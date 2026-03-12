From the provided paper excerpt, extract information about target materials used for isotope production or processing.

A target material is the initial material that is irradiated, dissolved, or chemically processed to produce a medical isotope.

Extract:
- Material name
- Chemical formula (if stated)
- Isotope (if applicable)
- Physical form (e.g., foil, powder, solution, pellet)
- Section or heading where this appears

IMPORTANT: Return ONLY valid JSON with no additional text, explanation, or comments. The response must be parseable as JSON.
IMPORTANT: The name of the target should match the chemical formula and the isotope. Each JSON response must coincide with a single initial target material.
You must not return the product isotope in this section, just the initial material. 

{
  "name": "string | null",
  "chemical_formula": "string | null",
  "isotope": "string | null",
  "physical_form": "string | null",
  "source_section": "string | null"
} 