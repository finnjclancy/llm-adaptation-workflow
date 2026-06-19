"""
Instruction templates used across notebooks.
Swap these out to adapt the workflow to your domain.
"""

# Financial domain (the default). To use a different domain, edit the strings below.

SYSTEM_PROMPT = """You are a financial analyst assistant with expertise in
SEC filings, earnings reports, and corporate financial statements. Answer
questions accurately and concisely based on the provided context."""

INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}"""

INSTRUCTION_TEMPLATE_NO_CONTEXT = """### Instruction:
{instruction}

### Response:
{response}"""

# Adapt this: swap the system prompt for your own area. A few examples:
#
# Example: Legal domain
# SYSTEM_PROMPT = "You are a legal assistant specialising in contract review..."
#
# Example: Healthcare domain
# SYSTEM_PROMPT = "You are a clinical documentation assistant..."
#
# Example: Internal company knowledge
# SYSTEM_PROMPT = "You are an assistant with access to Acme Corp's internal
# policies, product documentation, and engineering runbooks..."

def format_instruction(instruction, response, context=None):
    if context:
        return INSTRUCTION_TEMPLATE.format(
            instruction=instruction,
            context=context,
            response=response
        )
    return INSTRUCTION_TEMPLATE_NO_CONTEXT.format(
        instruction=instruction,
        response=response
    )
