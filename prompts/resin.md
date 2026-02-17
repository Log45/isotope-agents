From the provided paper excerpt, extract information about resins, chromatographic columns, or solid-phase extraction media.

Include materials used for separation, purification, or trapping of isotopes.

For each resin or column, extract:
- Name
- Base material (if stated)
- Mesh size or particle size (if stated)
- Column dimensions (if stated)
- Functional role in the process
- Section or heading where it appears

If multiple resins or columns are mentioned, return an array.

Output JSON matching this schema:
```json
{
  "resins_or_columns": [
    {
      "name": "string | null",
      "material": "string | null",
      "mesh_size": "string | null",
      "column_dimensions": "string | null",
      "role": "string | null",
      "source_section": "string | null"
    }
  ]
}
```