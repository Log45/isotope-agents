From the provided paper excerpt, extract information about target materials used for isotope production or processing.

A target material is the initial material that is irradiated, dissolved, or chemically processed to produce a medical isotope.

Extract:
- Material name
- Chemical formula (if stated)
- Isotope (if applicable)
- Physical form (e.g., foil, powder, solution, pellet)
- Section or heading where this appears

If multiple target materials are mentioned, return an array.

Output JSON matching this schema:

```json
{
  "target_materials": [
    {
      "name": "string | null",
      "chemical_formula": "string | null",
      "isotope": "string | null",
      "physical_form": "string | null",
      "source_section": "string | null"
    }
  ]
}
``` 