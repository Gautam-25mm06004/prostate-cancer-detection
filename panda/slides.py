"""Read bounded slide regions. Never decode an entire whole-slide TIFF."""
import heapq
from pathlib import Path
import numpy as np
from PIL import Image


class Slide:
    def __init__(self, path):
        path = Path(path)
        self.small = path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        if self.small:
            self.handle = Image.open(path)
            # PNG/JPEG decoding is not random-access; explicitly bound that fallback.
            if self.handle.width * self.handle.height > 16_000_000:
                self.handle.close()
                raise ValueError('Large PNG/JPEG: use the original tiled WSI TIFF instead.')
            self.handle.load()
            self.level_dimensions = [self.handle.size]
            self.level_downsamples = [1.0]
        else:
            import openslide
            self.handle = openslide.OpenSlide(str(path))
            self.level_dimensions = self.handle.level_dimensions
            self.level_downsamples = self.handle.level_downsamples
            try:
                self.handle.set_cache(openslide.OpenSlideCache(32 * 1024**2))
            except openslide.OpenSlideVersionError:
                pass

    def level(self, requested):
        value = requested if requested >= 0 else len(self.level_dimensions) + requested
        if not 0 <= value < len(self.level_dimensions):
            raise ValueError(f'Level {requested} unavailable; slide has {len(self.level_dimensions)} levels')
        return value

    def read(self, x, y, level, size, background=255):
        """x,y are always level-0 coordinates; size is in selected-level pixels."""
        if self.small:
            image = self.handle.convert('RGBA').crop((x, y, x + size, y + size))
        else:
            image = self.handle.read_region((int(x), int(y)), level, (size, size))
        canvas = Image.new('RGB', image.size, (background,) * 3)
        canvas.paste(image, mask=image.getchannel('A'))
        return np.asarray(canvas)

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def grid(slide, level, size, offset=0.0):
    """Centered padding matches the DrHB tiling grid; optional half-tile shift."""
    width, height = slide.level_dimensions[level]
    left = -((size - width % size) % size // 2) - int(size * offset)
    top = -((size - height % size) % size // 2) - int(size * offset)
    scale = slide.level_downsamples[level]
    for y in range(top, height, size):
        for x in range(left, width, size):
            yield round(x * scale), round(y * scale)


def darkest_coordinates(slide, level, size, count, offset=0.0):
    """Scan one tile at a time and keep only the top K coordinates, not pixels."""
    heap = []
    for index, (x, y) in enumerate(grid(slide, level, size, offset)):
        score = int(slide.read(x, y, level, size).sum(dtype=np.uint64))
        entry = (-score, -index, x, y)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [(x, y) for _, _, x, y in sorted(heap, reverse=True)]


def find_slide(folder, image_id, mask=False):
    if Path(image_id).name != image_id or image_id in {'.', '..'} or '\\' in image_id:
        raise ValueError(f'Invalid image_id: {image_id!r}')
    names = [image_id + '_mask', image_id] if mask else [image_id]
    for name in names:
        for suffix in ('.tiff', '.tif', '.svs', '.ndpi', '.png', '.jpg', '.jpeg'):
            path = Path(folder) / (name + suffix)
            if path.is_file():
                return path
    raise FileNotFoundError(f'No slide for {image_id} in {folder}')


def patch_target(mask, provider):
    mask = mask[..., 0]
    if provider == 'radboud':
        tumor = mask > 2
    elif provider == 'karolinska':
        tumor = mask == 2
    else:
        raise ValueError(f'Unknown data_provider: {provider}')
    return float(tumor.sum() > 100), float((mask > 0).any())
