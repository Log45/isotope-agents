from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata about the research paper."""
    title: str = Field(description="Paper title")
    doi: str = Field(description="Digital Object Identifier")
    source: str = Field(description="Publication source")


class TargetMaterial(BaseModel):
    """Target material information for isotope separation."""
    name: str = Field(description="Name of target material")
    chemical_formula: str = Field(description="Chemical formula")
    isotope: str = Field(description="Target isotope")
    physical_form: str = Field(description="Physical form of material")
    source_section: str = Field(description="Section reference in source document")


class AcidOrSolvent(BaseModel):
    """Information about acids, solvents, or bases used in the process."""
    name: str = Field(description="Name of substance")
    type: str = Field(description="Type: acid, solvent, or base")
    concentration: str = Field(description="Concentration specification")
    role: str = Field(description="Role: dissolution, wash, elution, or conditioning")
    source_section: str = Field(description="Section reference in source document")


class ResinOrColumn(BaseModel):
    """Information about resins or chromatographic columns."""
    name: str = Field(description="Name of resin or column")
    material: str = Field(description="Material composition")
    mesh_size: str = Field(description="Mesh size specification")
    column_dimensions: str = Field(description="Physical dimensions")
    role: str = Field(description="Role in separation process")
    source_section: str = Field(description="Section reference in source document")


class ElutionCondition(BaseModel):
    """Conditions for elution process."""
    eluent: str = Field(description="Eluent substance")
    concentration: str = Field(description="Concentration of eluent")
    volume: str = Field(description="Volume used")
    flow_rate: str = Field(description="Flow rate specification")
    temperature: str = Field(description="Temperature condition")
    pH: str = Field(description="pH value")
    source_section: str = Field(description="Section reference in source document")


class FinalProduct(BaseModel):
    """Information about the final product."""
    name: str = Field(description="Product name")
    isotope: str = Field(description="Isotope in final product")
    chemical_form: str = Field(description="Chemical form of product")
    purity: str = Field(description="Purity specification")
    yield_: str = Field(alias="yield", description="Yield percentage or amount")
    source_section: str = Field(description="Section reference in source document")

    class Config:
        populate_by_name = True


class IsotopeProcessFormat(BaseModel):
    """Complete structure for isotope separation process documentation."""
    paper_metadata: PaperMetadata = Field(description="Metadata about the research paper")
    target_materials: TargetMaterial = Field(description="Target material information")
    acids_and_solvents: AcidOrSolvent = Field(description="Acids, solvents, and bases used")
    resins_or_columns: ResinOrColumn = Field(description="Resins or columns used")
    elution_conditions: ElutionCondition = Field(description="Elution process conditions")
    final_products: FinalProduct = Field(description="Final product information")