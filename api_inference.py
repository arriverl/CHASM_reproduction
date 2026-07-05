import os
import json
import base64
import csv
import random
from datasets import load_dataset, disable_caching
from openai import OpenAI
from huggingface_hub import login
import getpass

def image_to_data_url(image_path):
    """Convert an image to data URL format"""
    with open(image_path, 'rb') as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        # Set MIME type based on image extension
        mime_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return f"data:{mime_type};base64,{base64_data}"

def call_openai_api(messages, model="gpt-4", api_key="", base_url=""):
    """
    Function to call the OpenAI API.

    Parameters:
        messages (list): List of chat messages
        model (str): Model name to use, e.g., "gpt-4"
        api_key (str): OpenAI API key
        base_url (str): Base URL for the API
    
    Returns:
        str: API response content
    """
    try:
        client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,  # Control randomness, 0 means deterministic output
            max_tokens=5000   # Maximum generation length
        )
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("API call error:", str(e))
        return None

def load_huggingface_dataset():
    """
    Load Hugging Face dataset
    
    Returns:
        dataset: Loaded dataset object
    """
    print("Preparing to load dataset...")
    
    # Disable caching
    disable_caching()
    
    # Try to use token from environment variables
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("\nNo Hugging Face Token found in environment variables")
        print("Please enter your Hugging Face Token to login (required to access the dataset):")
        token = getpass.getpass("HF Token: ")
    
    if token:
        print("Logging into Hugging Face with provided token...")
        login(token=token)
        print("Login successful!")
    else:
        print("No token provided, may not be able to access private datasets")
    
    dataset_name = "Jingyi77/CHASM-Covert_Advertisement_on_RedNote"
    
    try:
        print(f"Attempting to load dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        print(f"\nDataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
        
        return dataset
    except Exception as e:
        print(f"Error while trying to load dataset: {str(e)}")
        return None

def run_inference(dataset, model="gpt-4", sample_ratio=0.1, output_file="api_inference_results.csv"):
    """
    Run inference on the dataset and save results
    
    Parameters:
        dataset: Hugging Face dataset
        model (str): Model name to use
        sample_ratio (float): Sampling ratio, range 0-1
        output_file (str): CSV filename for output results
    """
    if not dataset:
        print("Dataset is empty, cannot run inference")
        return
    
    # Initialize statistics variables
    correct_predictions_0 = 0
    total_predictions_0 = 0
    correct_predictions_1 = 0
    total_predictions_1 = 0
    tn = fp = fn = tp = 0
    
    # Create CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Sample ID', 'Actual Label', 'Predicted Label', 'Text Content'])
        
        # Process each dataset split
        for split_name in dataset.keys():
            print(f"\nProcessing dataset split: {split_name}")
            split_data = dataset[split_name]
            
            # Determine sample count
            total_samples = len(split_data)
            sample_count = max(1, int(total_samples * sample_ratio))
            print(f"Total samples: {total_samples}, Will process: {sample_count} samples")
            
            # Randomly select sample indices
            sample_indices = random.sample(range(total_samples), sample_count)
            
            for idx in sample_indices:
                sample = split_data[idx]
                sample_id = sample.get('id', f"{split_name}_{idx}")
                actual_label = sample.get('label', -1)
                
                # Get text content
                title = sample.get('title', '')[:100]
                description = sample.get('description', '')[:200]
                
                # Get and merge comments
                comments = sample.get('comments', [])
                comments_text = ''
                if isinstance(comments, list):
                    comments_text = ' '.join([
                        comment.get('content', '') if isinstance(comment, dict) else str(comment)
                        for comment in comments
                    ])[:200]
                elif isinstance(comments, str):
                    comments_text = comments[:200]
                
                # Merge all text content
                post_text = f"{title} {description} {comments_text}".strip()
                
                # Check if text is empty
                if not post_text:
                    post_text = "No text content"
                
                # Get image path
                image_path = sample.get('image_path', None)
                
                # Build messages
                messages = [
                    {
                        "role": "system",
                        "content": "Characteristics of covert advertisements: 1. Clear promotional evidence: Covert ads typically contain obvious promotional traces, such as providing direct purchase links or product purchase instructions. To make the advertisement more covert, promotional links are sometimes embedded in images or comments, or users are redirected to private chat groups for sales. In contrast, non-advertising content primarily focuses on sharing personal experiences, so it may only casually mention product or store names, and the content usually lacks sufficient information for users to complete a purchase. 2. Post language style: Covert ads often use clickbait titles and sales pitches. These articles typically have a strong promotional tone, using exaggerated language to emphasize the advantages of products, which is contrary to the natural style of daily communication. In contrast, non-advertising content usually has a more casual tone, focusing on sharing personal experiences rather than promoting products. It may also mention product shortcomings. 3. Text and image structure of posts: Covert ads typically focus text and images on a single specific product or closely related products from the same brand. In contrast, non-promotional lifestyle sharing posts often involve multiple different brands in the same category, some of which may even be competitors, or the author may not explicitly recommend any specific brand. Your task is to determine whether social media posts contain advertising content. The input may include posts, images, and comments. If the input contains content with persuasive shopping tendencies, output '1' to indicate it contains advertisements. If the input is just general life-sharing content or other content unrelated to products, output '0'. Please only output '1'/'0', do not output any other content."
                    },
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                
                # Add text content
                messages[1]["content"].append({
                    "type": "text",
                    "text": post_text
                })
                
                # If there's an image, add image content
                if image_path and os.path.exists(image_path):
                    try:
                        image_data_url = image_to_data_url(image_path)
                        messages[1]["content"].insert(0, {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        })
                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                
                # Call API for inference
                try:
                    response = call_openai_api(messages=messages, model=model)
                    predicted_label = int(response) if response in ['0', '1'] else 0
                except Exception as e:
                    print(f"Error during inference: {str(e)}")
                    predicted_label = 0
                
                print(f"Sample {sample_id} - Actual label: {actual_label}, Predicted label: {predicted_label}")
                
                # Write results to CSV file
                writer.writerow([sample_id, actual_label, predicted_label, post_text[:50]])
                
                # Update statistics
                if actual_label == 0:
                    total_predictions_0 += 1
                    if predicted_label == 0:
                        correct_predictions_0 += 1
                        tn += 1
                    else:
                        fp += 1
                elif actual_label == 1:
                    total_predictions_1 += 1
                    if predicted_label == 1:
                        correct_predictions_1 += 1
                        tp += 1
                    else:
                        fn += 1
    
    # Print evaluation metrics
    print("\n=== Evaluation Results ===")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # Calculate metrics
    if total_predictions_0 > 0:
        accuracy_0 = correct_predictions_0 / total_predictions_0
        print(f"Non-ad classification accuracy: {accuracy_0 * 100:.2f}%")
    
    if total_predictions_1 > 0:
        accuracy_1 = correct_predictions_1 / total_predictions_1
        print(f"Ad classification accuracy: {accuracy_1 * 100:.2f}%")
    
    total = tp + tn + fp + fn
    if total > 0:
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Overall accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 score: {f1 * 100:.2f}%")
    
    # Save evaluation results
    results = {
        "accuracy": accuracy if 'accuracy' in locals() else None,
        "precision": precision if 'precision' in locals() else None,
        "recall": recall if 'recall' in locals() else None,
        "f1": f1 if 'f1' in locals() else None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }
    
    with open("api_inference_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation metrics saved to api_inference_metrics.json")
    print(f"Inference results saved to {output_file}")

def main():
    """Main function"""
    print("Starting API inference process...")
    
    # Load dataset
    dataset = load_huggingface_dataset()
    if not dataset:
        print("Unable to load dataset, program terminated")
        return
    
    # Choose model
    model_options = {
        "1": "gpt-4",
        "2": "gpt-4o-2024-08-06",
        "3": "gpt-4o-mini",
        "4": "gpt-3.5-turbo"
    }
    
    print("\nPlease select the model to use:")
    for key, value in model_options.items():
        print(f"{key}. {value}")
    
    model_choice = input("Enter option (default is 1): ").strip() or "1"
    model = model_options.get(model_choice, "gpt-4")
    
    # Set sampling ratio
    sample_ratio_input = input("\nEnter sampling ratio (between 0-1, default is 0.1): ").strip() or "0.1"
    try:
        sample_ratio = float(sample_ratio_input)
        if not (0 < sample_ratio <= 1):
            print("Sampling ratio out of range, using default value 0.1")
            sample_ratio = 0.1
    except ValueError:
        print("Invalid sampling ratio, using default value 0.1")
        sample_ratio = 0.1
    
    # Set output filename
    output_file = input("\nEnter output filename (default is api_inference_results.csv): ").strip() or "api_inference_results.csv"
    
    # Run inference
    print(f"\nRunning inference using model {model}, sampling ratio {sample_ratio}, results will be saved to {output_file}")
    run_inference(dataset, model=model, sample_ratio=sample_ratio, output_file=output_file)

if __name__ == "__main__":
    main()
