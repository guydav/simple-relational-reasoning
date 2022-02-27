from simple_relational_reasoning.datagen.object_fields import FIELD_TYPES
from simple_relational_reasoning.datagen.object_gen import ObjectGenerator, ObjectGeneratorDataset, \
    SpatialObjectGeneratorDataset
from simple_relational_reasoning.datagen.object_relations import OneDAdjacentRelation, MultipleDAdjacentRelation, \
    ColorAboveColorRelation, ObjectCountRelation, IdenticalObjectsRelation, BetweenRelation
from simple_relational_reasoning.datagen.quinn_objects import ObjectGeneratorWithSize, ObjectGeneratorWithoutSize, \
    DiagonalObjectGeneratorWithoutSize, QuinnBaseDatasetGenerator, QuinnWithReferenceDatasetGenerator, \
    CombinedQuinnDatasetGenerator, DiagonalAboveBelowDatasetGenerator, QuinnNoReferenceDatasetGenerator