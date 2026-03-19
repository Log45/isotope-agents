from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata about the research paper."""
    title: str | None = Field(default=None, description="Paper title")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    source: str | None = Field(default=None, description="Publication source")


class TargetMaterial(BaseModel):
    """Target material information for isotope separation."""
    name: str | None = Field(default=None, description="Name of target material")
    chemical_formula: str | None = Field(default=None, description="Chemical formula")
    isotope: str | None = Field(default=None, description="Target isotope")
    physical_form: str | None = Field(default=None, description="Physical form of material")
    source_section: str | None = Field(default=None, description="Section reference in source document")


class AcidOrSolvent(BaseModel):
    """Information about acids, solvents, or bases used in the process."""
    name: str | None = Field(default=None, description="Name of substance")
    type: str | None = Field(default=None, description="Type: acid, solvent, or base")
    concentration: str | None = Field(default=None, description="Concentration specification")
    role: str | None = Field(default=None, description="Role: dissolution, wash, elution, or conditioning")
    source_section: str | None = Field(default=None, description="Section reference in source document")


class ResinOrColumn(BaseModel):
    """Information about resins or chromatographic columns."""
    name: str | None = Field(default=None, description="Name of resin or column")
    material: str | None = Field(default=None, description="Material composition")
    mesh_size: str | None = Field(default=None, description="Mesh size specification")
    column_dimensions: str | None = Field(default=None, description="Physical dimensions")
    role: str | None = Field(default=None, description="Role in separation process")
    source_section: str | None = Field(default=None, description="Section reference in source document")


class ElutionCondition(BaseModel):
    """Conditions for elution process."""
    eluent: str | None = Field(default=None, description="Eluent substance")
    concentration: str | None = Field(default=None, description="Concentration of eluent")
    volume: str | None = Field(default=None, description="Volume used")
    flow_rate: str | None = Field(default=None, description="Flow rate specification")
    temperature: str | None = Field(default=None, description="Temperature condition")
    pH: str | None = Field(default=None, description="pH value")
    source_section: str | None = Field(default=None, description="Section reference in source document")


class FinalProduct(BaseModel):
    """Information about the final product."""
    name: str | None = Field(default=None, description="Product name")
    isotope: str | None = Field(default=None, description="Isotope in final product")
    chemical_form: str | None = Field(default=None, description="Chemical form of product")
    purity: str | None = Field(default=None, description="Purity specification")
    yield_: str | None = Field(default=None, alias="yield", description="Yield percentage or amount")
    source_section: str | None = Field(default=None, description="Section reference in source document")

    class Config:
        populate_by_name = True
        
class TargetMaterialList(BaseModel):
    items: list[TargetMaterial] = Field(default_factory=list)

class AcidOrSolventList(BaseModel):
    items: list[AcidOrSolvent] = Field(default_factory=list)

class ResinOrColumnList(BaseModel):
    items: list[ResinOrColumn] = Field(default_factory=list)

class ElutionConditionList(BaseModel):
    items: list[ElutionCondition] = Field(default_factory=list)

class FinalProductList(BaseModel):
    items: list[FinalProduct] = Field(default_factory=list)


class IsotopeProcessFormat(BaseModel):
    """Complete structure for isotope separation process documentation."""
    paper_metadata: PaperMetadata = Field(description="Metadata about the research paper")
    target_materials: TargetMaterialList = Field(description="Target material information")
    acids_and_solvents: AcidOrSolventList = Field(description="Acids, solvents, and bases used")
    resins_or_columns: ResinOrColumnList = Field(description="Resins or columns used")
    elution_conditions: ElutionConditionList = Field(description="Elution process conditions")
    final_products: FinalProductList = Field(description="Final product information")