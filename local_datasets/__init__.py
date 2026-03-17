import os
from .refcoco import RefDataset
from .refcoco_adv import RefCOCOAdvDataset
from .clevr import CLEVRDataset


def get_dataset(name, num_samples=None):
    """
    Get a dataset by name.

    Dataset paths are read from environment variables. Set these in your .env file
    or shell environment before running:

        MSCOCO_DIR        — path to MS-COCO images root (used by RefCOCO variants & RefCOCO-Adv)
        REFCOCO_DIR       — path to RefCOCO annotation root (contains refcoco/, refcoco+/, refcocog/, refgta/)
        GTA_IMAGE_DIR     — path to RefGTA images (if separate from REFCOCO_DIR/refgta/)
        REFCOCO_ADV_FILE  — full path to refcocog_adv_annotations.json
        CLEVR_DIR         — path to CLEVR dataset root (contains val/questions.json, val/scenes.json, val/images/)
        CLEVR_HUMANS_DIR  — path to CLEVR-Humans dataset root (contains val/questions.json, val/scenes.json)
                            Uses CLEVR_DIR/val/images/ for images.
    """
    MSCOCO_DIR       = os.getenv("MSCOCO_DIR", "data/mscoco")
    REFCOCO_DIR      = os.getenv("REFCOCO_DIR", "data/refcoco")
    GTA_IMAGE_DIR    = os.getenv("GTA_IMAGE_DIR", os.path.join(REFCOCO_DIR, "refgta"))
    REFCOCO_ADV_FILE = os.getenv(
        "REFCOCO_ADV_FILE",
        os.path.join(REFCOCO_DIR, "refcocog-adv", "refcocog_adv_annotations.json"),
    )
    CLEVR_DIR        = os.getenv("CLEVR_DIR", "data/clevr")
    CLEVR_HUMANS_DIR = os.getenv("CLEVR_HUMANS_DIR", "data/clevr-humans")

    if name == "refcoco-tune":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcocog'),
            split='test',
            refs_filename='refs(umd).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples or 100,
        )

    if name == "refcocog":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcocog'),
            split='test',
            refs_filename='refs(umd).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "refcoco+val":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcoco+'),
            split='val',
            refs_filename='refs(unc).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "refcoco+A":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcoco+'),
            split='testA',
            refs_filename='refs(unc).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "refcoco+B":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcoco+'),
            split='testB',
            refs_filename='refs(unc).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "refcocoval":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcoco'),
            split='val',
            refs_filename='refs(unc).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "refcocoA":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcoco'),
            split='testA',
            refs_filename='refs(unc).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "refcocoB":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refcoco'),
            split='testB',
            refs_filename='refs(unc).p',
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "refgta":
        return RefDataset(
            annotation_path=os.path.join(REFCOCO_DIR, 'refgta'),
            split='test',
            refs_filename='refs(utokyo).p',
            image_dir=GTA_IMAGE_DIR,
            num_samples=num_samples or 2000,
        )

    elif name == 'refcoco-adv':
        return RefCOCOAdvDataset(
            annotation_file=REFCOCO_ADV_FILE,
            image_dir=MSCOCO_DIR,
            num_samples=num_samples,
        )

    elif name == "clevr-humans":
        clevr_images = os.path.join(CLEVR_DIR, "val", "images")
        return CLEVRDataset(
            questions_path=os.path.join(CLEVR_HUMANS_DIR, "val", "questions.json"),
            scenes_path=os.path.join(CLEVR_HUMANS_DIR, "val", "scenes.json"),
            images_dir=clevr_images,
            num_samples=num_samples,
        )
    elif name == "clevr":
        return CLEVRDataset(
            questions_path=os.path.join(CLEVR_DIR, "val", "questions.json"),
            scenes_path=os.path.join(CLEVR_DIR, "val", "scenes.json"),
            images_dir=os.path.join(CLEVR_DIR, "val", "images"),
            num_samples=num_samples,
        )
    else:
        raise ValueError(f"Dataset '{name}' not recognized.")
