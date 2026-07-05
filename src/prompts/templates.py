"""论文 Appendix D 提示词模板。"""

ZERO_SHOT_PROMPT = """Your task is to determine whether a social media post contains advertising content. The input may include tweets, images, and comments. If the input contains persuasive content encouraging shopping, output '1' to indicate the presence of an advertisement. If the input is just general life-sharing content or unrelated to products, output '0'. Please output only '1' or '0' without any additional text."""

FEW_SHOT_PROMPT_HEADER = ZERO_SHOT_PROMPT

DETAILED_GUIDELINES = """
Here are some guidelines: 1. Clear evidence of promotion: Hidden ads often contain obvious signs of promotion, such as providing direct purchase links or product purchase instructions. To make the ads more hidden, promotional links are sometimes embedded in pictures or comments, or users are redirected to private chat groups for sales. In contrast, non-advertising content focuses mainly on sharing personal experiences, so it may only casually mention product or store names, and the content usually lacks enough information for users to complete the purchase. 2. Post language style: Hidden ads often use clickbait-style titles and sales pitches. Such articles often have a strong promotional tone and use exaggerated language to emphasize the advantages of the product, which runs counter to the natural style of daily communication. In contrast, non-advertising content is usually more casual in tone and focuses on sharing personal experiences rather than promoting products. It may also mention product shortcomings. 3. Post text and image structure: Hidden ads often focus text and images on a single specific product or closely related products of the same brand. In contrast, non-promotional lifestyle sharing posts often involve multiple different brands in the same category, some of which may even be competitors, or the author does not explicitly recommend any specific brand."""

DETAILED_ZERO_SHOT_PROMPT = (
    "Your task is to determine whether the social media tweets contain advertising content. "
    "The input may include tweets, pictures, and comments. If the input contains content that persuades "
    "people to buy, the output is '1', which means it contains advertising. If the input is just general "
    "life sharing content or other content not related to the product, the output is '0'. Please only "
    "output '1'/'0', and do not output other content."
    + DETAILED_GUIDELINES
)

DETAILED_FEW_SHOT_PROMPT_HEADER = DETAILED_ZERO_SHOT_PROMPT

SYSTEM_GUIDELINES_EN = (
    "Characteristics of covert advertisements: "
    "1. Clear promotional evidence: Covert ads typically contain obvious promotional traces, such as "
    "providing direct purchase links or product purchase instructions. To make the advertisement more "
    "covert, promotional links are sometimes embedded in images or comments, or users are redirected to "
    "private chat groups for sales. In contrast, non-advertising content primarily focuses on sharing "
    "personal experiences, so it may only casually mention product or store names, and the content "
    "usually lacks sufficient information for users to complete a purchase. "
    "2. Post language style: Covert ads often use clickbait titles and sales pitches. These articles "
    "typically have a strong promotional tone, using exaggerated language to emphasize the advantages "
    "of products, which is contrary to the natural style of daily communication. In contrast, "
    "non-advertising content usually has a more casual tone, focusing on sharing personal experiences "
    "rather than promoting products. It may also mention product shortcomings. "
    "3. Text and image structure of posts: Covert ads typically focus text and images on a single "
    "specific product or closely related products from the same brand. In contrast, non-promotional "
    "lifestyle sharing posts often involve multiple different brands in the same category, some of "
    "which may even be competitors, or the author may not explicitly recommend any specific brand."
)


def get_prompt(mode: str = "zero_shot") -> str:
    mapping = {
        "zero_shot": ZERO_SHOT_PROMPT,
        "in_context": FEW_SHOT_PROMPT_HEADER,
        "detailed_zero_shot": DETAILED_ZERO_SHOT_PROMPT,
        "detailed_in_context": DETAILED_FEW_SHOT_PROMPT_HEADER,
    }
    if mode not in mapping:
        raise ValueError(f"Unknown prompt mode: {mode}. Choose from {list(mapping)}")
    return mapping[mode]


def format_post_text(
    title: str,
    description: str,
    comments,
    include_comments: bool = True,
) -> str:
    parts = [title or "", description or ""]
    if include_comments and comments:
        if isinstance(comments, list):
            comment_texts = []
            for c in comments:
                if isinstance(c, dict):
                    comment_texts.append(c.get("content", str(c)))
                else:
                    comment_texts.append(str(c))
            parts.append(" ".join(comment_texts))
        elif isinstance(comments, str):
            parts.append(comments)
    return "\n".join(p.strip() for p in parts if p and p.strip())


def build_few_shot_prompt(
    header: str,
    positive_example: str,
    negative_example: str,
    query_text: str,
) -> str:
    return (
        f"{header}\n\n"
        f"[A Selected Convert Advertisement Example]\n{positive_example}\n\n"
        f"[A Selected Non-Convert Advertisement Example]\n{negative_example}\n\n"
        f"[Query Post]\n{query_text}"
    )
