From the provided paper excerpt, extract all acids, bases, and solvents used in chemical processing steps.

Include chemicals used for:
- Dissolution
- Washing
- Elution
- Column conditioning

For each chemical, extract:
- Name
- Type (acid, base, or solvent)
- Concentration (if stated)
- Role in the process
- Section or heading where it appears

IMPORTANT: Return ONLY valid JSON with no additional text, explanation, or comments. The response must be parseable as JSON.

{
  "items": [
    {
      "name": "string | null",
      "type": "acid | solvent | base | null",
      "concentration": "string | null",
      "role": "dissolution | wash | elution | conditioning | null",
      "source_section": "string | null"
    }
  ]
}