from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    classes: tuple[str, ...]
    prompts: tuple[str, ...]
    has_dsm: bool
    ignore_index: int = 255

ISPRS_CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car')
ISPRS_PROMPTS = (
    'an aerial image of impervious surfaces.',
    'an aerial image of buildings.',
    'an aerial image of low vegetation.',
    'an aerial image of trees.',
    'an aerial image of cars.',
)
LOVEDA_CLASSES = ('background','building','road','water','barren','forest','agriculture')
LOVEDA_PROMPTS = (
    'a satellite image of miscellaneous background regions.',
    'a satellite image of buildings.',
    'a satellite image of roads.',
    'a satellite image of water.',
    'a satellite image of barren land.',
    'a satellite image of forests.',
    'a satellite image of agricultural land.',
)
UAVID_CLASSES = ('clutter','building','road','tree','low_vegetation','moving_car','static_car','human')
UAVID_PROMPTS = (
    'an aerial image of miscellaneous objects and clutter in an urban scene.',
    'an aerial image of buildings.',
    'an aerial image of roads.',
    'an aerial image of trees.',
    'an aerial image of low vegetation.',
    'an aerial image of moving cars.',
    'an aerial image of parked cars.',
    'an aerial image of people.',
)
DATASETS = {
    'potsdam': DatasetConfig('potsdam', ISPRS_CLASSES, ISPRS_PROMPTS, True),
    'vaihingen': DatasetConfig('vaihingen', ISPRS_CLASSES, ISPRS_PROMPTS, True),
    'loveda': DatasetConfig('loveda', LOVEDA_CLASSES, LOVEDA_PROMPTS, False),
    'uavid': DatasetConfig('uavid', UAVID_CLASSES, UAVID_PROMPTS, False),
}

def get_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset: {name}. Valid: {list(DATASETS)}')
    return DATASETS[name]
