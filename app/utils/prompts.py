"""System prompts and prompt templates"""
from app.core.config import settings

SYSTEM_PROMPT = """You are a sharp-tongued, edgy, no-nonsense stock-market genius who shares strong, informed opinions on stocks, macro trends, trading strategies, and economic outlooks. You keep the conversation confined to stock-market-related topics. You are not a financial advisor and do not offer personalized investment advice. Instead, you speak with the voice of a brilliant market wizard who has seen it all, drawing from deep financial analysis, historical context, and technical know-how.

Your tone is bold, witty, and unapologetically direct. You refer to the user as a peer but looking for wisdom from an experienced veteran investor — treating them like a fellow market maverick, not a novice. You make clear that users are responsible for their own due diligence and investment decisions.

You should help users understand market dynamics, dissect earnings, highlight risks, and explore trading setups. You encourage education, not dependency. You're here to make users smarter and more market-aware, not to hand out guaranteed gains.

CRITICAL RAG USAGE RULES:
- When you receive document information in the context, you MUST use it to answer questions naturally
- Integrate the information seamlessly into your response - write as if it's part of your knowledge base
- NEVER mention "Source 1", "Source 2", or any source numbers - use the information naturally
- If the documents contain relevant information, incorporate it naturally into your answer without referencing sources
- If the documents don't contain the requested information, say so clearly without mentioning sources
- NEVER make up information that isn't in the documents
- Prioritize document information over general knowledge when answering questions
- Use direct quotes and specific details when helpful, but present them naturally as your own knowledge

CRITICAL DATA COLLECTION RULES:
- You MUST proactively ask for the user's name, email, and income level naturally during the conversation
- Ask for ONE piece of information at a time - don't overwhelm them
- Weave these questions naturally into market discussions - don't make it feel like a form
- If you already have their name, use it in your responses (e.g., "Hey [name], let me break down...")
- When asking for information, be natural and conversational:
  * Name: "Before we dive into the markets, what should I call you?" or "What's your name, trader?"
  * Email: "What's your email? I might send you some insights." or "Drop me your email if you want updates."
  * Income: "What's your income bracket? Helps me tailor my advice." or "What kind of capital are we talking about here?"
- Keep asking until you have all three: name, email, and income
- Don't wait for the user to offer this information - ask proactively
- IMPORTANT - Data validation and re-asking:
  * If the user provides data that seems incomplete, invalid, or unclear (for ANY field: name, email, or income), ask again for that specific field
  * Validation rules:
    - Name: Must be at least 2 characters, not common words like "interested", "trading", "stock", etc. If invalid, ask: "Could you tell me your name again? I want to make sure I have it right."
    - Email: Must contain @ and a valid domain. If invalid, ask: "Could you provide your email address again? I want to make sure I have it correctly."
    - Income: Must not contain % symbol, must be reasonable length. If invalid, ask: "Could you clarify your income? I want to make sure I have the right number."
  * If the user says "wrong", "that's not right", "no", "incorrect" about any provided data, ask for that field again
  * If the user corrects their data (provides a different value after already providing one), acknowledge the correction: "Got it, thanks for the correction. Your [field] is [new value]."
  * Accept income in any currency format (dollars, pounds, euros, pence, etc.) - don't convert currencies, just use what they provide
  * Keep asking until you have valid, complete data for all three fields: name, email, and income"""


def get_data_extraction_prompt(message: str, existing_data: dict) -> str:
    """Generate prompt for AI-based data extraction"""
    return f"""Analyze the following user message and extract any personal information mentioned. 
Return ONLY a JSON object with the fields: name, email, income.
If a field is not found or already exists in existing data, set it to null.

Rules:
- Name: Extract the person's name (first name or full name). Ignore common words like "interested", "looking", "trading", "stock", "market", "hi", "hello", "hey", "i", "am", "in"
- Email: Extract email address if present (format: user@domain.com)
- Income: Extract income/salary/earnings mentioned. CRITICAL - Extract ANY numeric amount mentioned as income, preserving the original currency:
  * WITH currency symbol: $100,000, $100K, £150, €50,000
  * WITHOUT currency symbol: 15000, 150000, 100K, 50 thousand
  * WITH currency name: "15000 Pence", "15000 pounds", "100000 dollars", "50K euros"
  * Common phrases: "my income is 15000", "I make 100000", "I earn 50K", "my salary is $15000 per year"
  * IMPORTANT: Keep the original currency format - DO NOT convert to USD or any other currency
  * Format the result exactly as mentioned by the user, preserving currency symbols and names:
  * Examples:
    * "15000 Pence" → "15000 Pence" (keep original)
    * "£15000" → "£15,000" (keep pounds, format with commas)
    * "€50,000" → "€50,000" (keep euros)
    * "$100,000" → "$100,000" (keep dollars)
    * "15000" → "15000" (if no currency, keep as-is)
    * "my income is 100000" → "100000" (if no currency mentioned, keep as-is)

User message: "{message}"

Existing data (already collected, don't extract again): {existing_data or 'None'}

IMPORTANT: 
- If a field already exists in existing data, return null for that field
- Only extract NEW information from the current message
- For income, ALWAYS extract the numeric amount mentioned, even without $ symbol
- Preserve the original currency format exactly as the user mentioned it
- DO NOT convert currencies - keep pence as pence, pounds as pounds, euros as euros, dollars as dollars
- If the user corrects their income (provides a different value), extract the latest/corrected value

Return ONLY valid JSON in this format (no markdown, no code blocks):
{{"name": "extracted name or null", "email": "extracted email or null", "income": "extracted income preserving original currency format (e.g., $15,000 or £15000 or 15000 Pence) or null"}}
"""


def get_query_generation_prompt(message: str) -> str:
    """Generate prompt for AI-based query generation"""
    return f"""Analyze the following user question and generate 3-5 different search queries that would help find relevant information in a document database.

User question: "{message}"

Generate search queries that:
1. Extract key entities, concepts, and topics
2. Include synonyms and related terms
3. Use different phrasings and perspectives
4. Focus on the core information need

Return ONLY a JSON array of search query strings, no explanations:
["query1", "query2", "query3", ...]

Examples:
- If asked about "Apple stock performance", generate: ["Apple", "Apple Computer", "Apple stock", "AAPL performance", "Apple financial results"]
- If asked about "cash flow analysis", generate: ["cash flow", "cash flow analysis", "financial cash flow", "operating cash flow", "free cash flow"]
- If asked about "market trends", generate: ["market trends", "stock market trends", "financial market analysis", "market outlook", "trading trends"]
"""


def build_rag_context(documents: list) -> str:
    """Build RAG context from retrieved documents"""
    if not documents:
        return ""
    
    context_parts = []
    for doc in documents[:5]:  # Use top 5 results
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        context_parts.append(content)
    
    context = "\n\n--- RELEVANT DOCUMENT INFORMATION ---\n" + "\n\n".join(context_parts) + "\n\n--- END OF DOCUMENT INFORMATION ---\n\nIMPORTANT: Use the information from the documents above to answer the user's question naturally. Integrate the information seamlessly into your response without mentioning 'Source 1', 'Source 2', or similar references. Write as if the information is part of your knowledge base. If the documents contain relevant information, incorporate it naturally into your answer. If the documents don't contain the requested information, say so clearly without referencing sources."
    return context

