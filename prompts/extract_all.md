You are given a **summary of a research paper** describing isotope production and chemical separation.

Your task is to extract all relevant information into a structured format matching the required schema.

---

### Instructions

From the provided summary, extract information for the following categories:

* Paper metadata
* Target materials
* Acids, solvents, and bases
* Resins or columns
* Elution conditions
* Final products

---

### Extraction Rules

* Extract only information explicitly present in the summary
* Do NOT infer or guess missing values
* If a field is not specified, return `null`
* Preserve all values exactly as written (including units, formatting, and notation)
* If multiple items exist, include all of them
* Keep text concise but faithful to the source

---

### Field Guidance

#### Paper Metadata

* Title of the paper (if present)
* DOI (if present)
* Journal or source (if present)

#### Target Materials

* Material name
* Chemical formula
* Isotope (if specified)
* Physical form (foil, powder, solution, etc.)
* Section reference if available

#### Acids and Solvents

* Name of chemical
* Type (acid, solvent, base)
* Concentration (e.g., 1 M, 0.1 M, etc.)
* Role (e.g., dissolution, wash, elution, conditioning)

#### Resins or Columns

* Name (e.g., DGA, AG1-X8)
* Material (if stated)
* Mesh size
* Column dimensions
* Role in separation

#### Elution Conditions

* Eluent (chemical used)
* Concentration
* Volume
* Flow rate
* Temperature
* pH

#### Final Products

* Product name
* Isotope
* Chemical form
* Purity
* Yield

---

### Output Requirements

* Return ONLY valid JSON
* No explanations, no extra text
* Must match this exact structure:

```json
{
  "paper_metadata": {
    "title": "string | null",
    "doi": "string | null",
    "source": "string | null"
  },
  "target_materials": {
    "items": [
      {
        "name": "string | null",
        "chemical_formula": "string | null",
        "isotope": "string | null",
        "physical_form": "string | null",
        "source_section": "string | null"
      }
    ]
  },
  "acids_and_solvents": {
    "items": [
      {
        "name": "string | null",
        "type": "string | null",
        "concentration": "string | null",
        "role": "string | null",
        "source_section": "string | null"
      }
    ]
  },
  "resins_or_columns": {
    "items": [
      {
        "name": "string | null",
        "material": "string | null",
        "mesh_size": "string | null",
        "column_dimensions": "string | null",
        "role": "string | null",
        "source_section": "string | null"
      }
    ]
  },
  "elution_conditions": {
    "items": [
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
  },
  "final_products": {
    "items": [
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
}
```