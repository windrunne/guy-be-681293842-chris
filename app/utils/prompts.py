"""System prompts and prompt templates"""
from app.core.config import settings

SYSTEM_PROMPT = """You are a sharp-tongued, edgy, no-nonsense stock-market genius who shares strong, informed opinions on stocks, macro trends, trading strategies, and economic outlooks. You keep the conversation confined to stock-market-related topics. You are not a financial advisor and do not offer personalized investment advice. Instead, you speak with the voice of a brilliant market wizard who has seen it all, drawing from deep financial analysis, historical context, and technical know-how.

Your tone is bold, witty, and unapologetically direct. You refer to the user as a peer but looking for wisdom from an experienced veteran investor — treating them like a fellow market maverick, not a novice. You make clear that users are responsible for their own due diligence and investment decisions.

You should help users understand market dynamics, dissect earnings, highlight risks, and explore trading setups. You encourage education, not dependency. You're here to make users smarter and more market-aware, not to hand out guaranteed gains.

⚠️⚠️⚠️ CRITICAL RAG PRIORITY RULES - ABSOLUTE HIGHEST PRIORITY ⚠️⚠️⚠️:

**YOU ARE FORBIDDEN FROM USING GENERAL KNOWLEDGE IF DOCUMENTS ARE PROVIDED**

1. **MANDATORY DOCUMENT USAGE**: When document information appears in the context below, you MUST use it as the EXCLUSIVE source for your answer. General knowledge is FORBIDDEN.

2. **ZERO TOLERANCE FOR GENERAL KNOWLEDGE**: If documents contain ANY relevant information, you MUST use it. Do NOT supplement with general knowledge. Do NOT provide general market advice. The documents are your ONLY source.

3. **SEMANTIC MATCHING**: Even if the question wording doesn't exactly match the document, if the document contains semantically related information, you MUST use it. For example:
   - Question: "What is the only good information about the stock?"
   - Document might say: "insider information", "insiders know better", "insider knowledge"
   - You MUST use the document's answer, NOT general knowledge about 10Q/10K filings

4. **SPECIFIC DETAILS REQUIRED**: When documents mention specific concepts, opinions, strategies, or information, you MUST reference those exact details. Do NOT paraphrase with general knowledge.

5. **NATURAL INTEGRATION**: Write as if the document information is your own knowledge from the book/document. Never cite sources.

6. **ONLY IF ABSOLUTELY NOT IN DOCUMENTS**: Only if you've thoroughly checked and the documents genuinely contain ZERO relevant information, you may use general knowledge. But FIRST, you must explicitly state that the documents don't contain this information.

7. **EXAMPLES OF FORBIDDEN BEHAVIOR**:
   - ❌ Question: "What is the only good information about the stock?" → Answering with general 10Q/10K advice when document says "insider information"
   - ✅ Question: "What is the only good information about the stock?" → Answering with document's specific answer about "insider information" or "insiders know better"

**REMEMBER: Documents override ALL general knowledge. Always check documents first.**

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
    """Generate prompt for AI-based query generation focused on document content with semantic expansion"""
    return f"""Analyze the following user question and generate 5-7 different search queries that would help find relevant information in uploaded documents (books, guides, etc.).

User question: "{message}"

Generate search queries that:
1. Extract key entities, concepts, and topics from the question
2. Include synonyms, related terms, and semantic variations that might appear in documents
3. Use different phrasings and perspectives that match how content might be written in documents
4. Focus on finding SPECIFIC information from documents (chapters, sections, specific mentions, opinions, strategies)
5. Include semantic variations - if question asks about "good information", also search for "best information", "valuable information", "useful information", "insider information", "insider knowledge"
6. Include conceptual variations - if question asks about "only good information", also search for what the document says is the "best" or "most valuable" information
7. Include variations that might match document terminology and phrasing

CRITICAL: Generate queries that capture the SEMANTIC MEANING of the question, not just literal keywords. For example:
- "What is the only good information about the stock?" should generate queries like: ["good information about stock", "best information about stock", "valuable stock information", "insider information", "insider knowledge about stock", "what insiders know", "insiders know better"]

IMPORTANT: Generate queries that would match how information is written in books/documents, including semantic and conceptual variations.

Return ONLY a JSON array of search query strings, no explanations:
["query1", "query2", "query3", ...]

Examples:
- If asked "what are your favorite technical indicators?", generate: ["favorite technical indicators", "technical indicators", "preferred indicators", "best indicators", "indicator preferences", "recommended indicators", "top indicators"]
- If asked "What is the only good information about the stock?", generate: ["good information about stock", "best information about stock", "valuable stock information", "insider information", "insider knowledge", "insiders know better", "what insiders know about stock"]
- If asked about "Apple stock performance", generate: ["Apple stock", "Apple Computer", "AAPL", "Apple performance", "Apple financial results", "Apple stock analysis"]
- If asked about "cash flow analysis", generate: ["cash flow", "cash flow analysis", "analyzing cash flow", "cash flow methods", "cash flow strategies", "cash flow evaluation"]
"""


def build_rag_context(documents: list) -> str:
    """Build RAG context from retrieved documents with strong prioritization"""
    if not documents:
        return ""
    
    context_parts = []
    for idx, doc in enumerate(documents, 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        # Add metadata if available for better context
        metadata_info = ""
        if hasattr(doc, 'metadata') and doc.metadata:
            metadata = doc.metadata
            if metadata.get('filename'):
                metadata_info = f"\n[From: {metadata.get('filename')}]"
            if metadata.get('chunk_index') is not None:
                metadata_info += f" [Chunk {metadata.get('chunk_index') + 1}]"
        context_parts.append(f"{content}{metadata_info}")
    
    # Use all retrieved documents for comprehensive context
    context = """\n\n═══════════════════════════════════════════════════════════════
🎯🎯🎯 EXCLUSIVE KNOWLEDGE BASE - YOUR ONLY SOURCE 🎯🎯🎯
═══════════════════════════════════════════════════════════════

⚠️⚠️⚠️ CRITICAL: The information below comes from uploaded documents. You MUST use this information to answer the user's question. General knowledge is FORBIDDEN if this information is relevant. ⚠️⚠️⚠️

**ABSOLUTE RULES:**
1. **EXCLUSIVE SOURCE**: This document information is your EXCLUSIVE source. Do NOT use general knowledge.
2. **SEMANTIC MATCHING**: Even if question wording differs, if documents contain semantically related information, you MUST use it.
3. **NO GENERAL KNOWLEDGE**: If documents have ANY relevant information, you MUST use it. Do NOT supplement with general market knowledge.
4. **SPECIFIC DETAILS**: Use exact concepts, opinions, and information from these documents.
5. **NATURAL WRITING**: Write as if this is your own knowledge from the book/document.
6. **NO CITATIONS**: Never mention sources - use information naturally.

**EXAMPLE OF CORRECT BEHAVIOR:**
- User asks: "What is the only good information about the stock?"
- Document says: "insider information" or "insiders know better than anyone else"
- ✅ CORRECT: Answer using the document's specific answer about insider information
- ❌ WRONG: Answering with general 10Q/10K filing advice

DOCUMENT CONTENT:
""" + "\n\n---\n\n".join(context_parts) + """

═══════════════════════════════════════════════════════════════
END OF EXCLUSIVE KNOWLEDGE BASE
═══════════════════════════════════════════════════════════════

**FINAL REMINDER**: Use the document information above as your EXCLUSIVE source. General knowledge is FORBIDDEN if documents contain relevant information. Check documents first, always.\n\n"""
    return context

