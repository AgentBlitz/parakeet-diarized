To get the absolute best extraction results out of the Granite 3.3 8B Instruct model (including its quantized INT versions), you should leverage its native support for structured reasoning using `<think>` and `<response>` tags.[1] Because data extraction and summarization are precision tasks, IBM recommends setting your decoding parameters to "Greedy" (e.g., a temperature of 0) so the model returns highly predictable and accurate content rather than "creative" hallucinations.

Here is the optimal way to set up the system prompt, user prompt, and generation parameters for your diarized transcripts.

### 1\. System Prompt

The system prompt needs to explicitly enforce the two-stage cognitive process and define the exact JSON schema you want it to output.

You are an expert AI meeting analyst. Your task is to analyze a diarized meeting transcript and extract structured information.

You must strictly follow this two-step process:

1.  First, use a \<think\> block to internally analyze the transcript. Within this block, map out who is speaking, track commitments across the conversation to resolve pronouns to specific speakers, and evaluate whether questions were answered or left open.
2.  Second, use a \<response\> block to output a valid JSON object containing the final extracted data. Do not output any text outside of the \<think\> and \<response\> blocks.

The JSON object in the \<response\> block must exactly match this schema:
{
"summary": "A concise, abstractive paragraph summarizing the macro-level narrative and key decisions.",
"action\_items":,
"unresolved\_questions": [
"List of strategic or open questions asked but not definitively answered during the meeting."
]
}

### 2\. User Prompt

Pass the raw transcript wrapped in clear text delimiters (like XML tags) so the model knows exactly where the conversational data begins and ends.

Please analyze the following meeting transcript and extract the required features:

\<transcript\>
Speaker 1 (John): Let's review the server migration. Sarah, can you handle the database backups by Friday?
Speaker 2 (Sarah): Yeah, I'll take care of it before the weekend. Did we ever figure out the licensing costs for the new monitoring tool?
Speaker 1 (John): Not yet, I need to email the vendor about that.
\</transcript\>

### 3\. Generation Parameters

When querying the model through your inference engine (such as vLLM, Ollama, or LM Studio), you should configure the following parameters:

  * **Decoding Strategy:** Set decoding to Greedy (or set `temperature` to `0.0` and `top_p` to `1.0`). This forces the model to choose the most logically probable tokens, which is critical for strict JSON adherence and factual extraction.
  * **Max New Tokens:** Because the model will generate a potentially lengthy internal reasoning path inside the `<think>` tags before generating your JSON, ensure your max tokens limit is set high enough (e.g., 4000+ tokens) so the output isn't cut off mid-generation.

By forcing the model into the `<think>` block first, it will silently map "I'll take care of it" to "Sarah" and recognize that the licensing cost question was left unresolved, guaranteeing that the final `<response>` JSON contains perfectly extracted business intelligence.[1]