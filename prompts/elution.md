From the provided paper excerpt, extract elution conditions used during chromatographic or column-based separation.

Elution conditions include chemical and physical parameters explicitly associated with the elution step.

Extract:
- Eluent name
- Eluent concentration
- Elution volume
- Flow rate
- Temperature
- pH
- Section or heading where this appears

If multiple elution steps are described, return an array.

Output JSON matching this schema:
```json
{
  "elution_conditions": [
    {
      "eluent": "string | null",
      "concentration": "string | null",
      "volume": "string | null",
      "flow_rate": "string | null",
      "temperature": "string | null",
      "pH": "string | null",
      "source_section": "string | null"
    }
  ]
}
```