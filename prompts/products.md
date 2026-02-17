From the provided paper excerpt, extract information about the final radioactive or chemical products produced at the end of the process.

Final products include purified isotopes or compounds intended for medical, analytical, or research use.

For each final product, extract:
- Product name
- Isotope
- Chemical form
- Reported purity
- Reported yield
- Section or heading where this appears

If multiple final products are mentioned, return an array.

Output JSON matching this schema:
```json
{
  "final_products": [
    {
      "name": "string | null",
      "isotope": "string | null",
      "chemical_form": "string | null",
      "purity": "string | null",
      "yield": "string | null",
      "source_section": "string | null"
    }
  ]
}
```