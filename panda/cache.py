"""Bounded per-slide NPY or JPEG-in-ZIP tile storage."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image


class TileWriter:
    def __init__(self, folder, config):
        self.folder, self.config = Path(folder), config
        self.compressed = config.cache_format == 'jpeg'
        self.path = self.folder / ('tiles.zip' if self.compressed else 'tiles.npy')
        self.temporary = self.path.with_name(self.path.stem + '.partial' + self.path.suffix)
        self.count = 0

    def __enter__(self):
        c = self.config
        if self.compressed:
            self.handle = zipfile.ZipFile(self.temporary, 'w', compression=zipfile.ZIP_STORED)
        else:
            self.handle = np.lib.format.open_memmap(self.temporary, mode='w+', dtype=np.uint8,
                                                   shape=(c.cache_tiles, c.tile_size, c.tile_size, 3))
        return self

    def write(self, index, tile):
        if index != self.count or tile.shape != (self.config.tile_size, self.config.tile_size, 3):
            raise ValueError('Write correctly shaped tiles in consecutive index order')
        if self.compressed:
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(tile, dtype=np.uint8)).save(buffer, format='JPEG',
                        quality=self.config.jpeg_quality, subsampling=0)
            self.handle.writestr(f'{index:04d}.jpg', buffer.getvalue())
        else:
            self.handle[index] = tile
        self.count += 1

    def __exit__(self, error_type, *_):
        if self.compressed:
            self.handle.close()
        else:
            self.handle.flush()
        del self.handle
        if error_type is None:
            if self.count != self.config.cache_tiles:
                raise ValueError('Incomplete tile cache')
            self.temporary.replace(self.path)


def read_tiles(folder, metadata, count):
    folder = Path(folder)
    if metadata.get('cache_format', 'npy') == 'npy':
        tiles = np.load(folder/'tiles.npy', mmap_mode='r', allow_pickle=False)
        if tiles.shape != (metadata['cache_tiles'], metadata['tile_size'], metadata['tile_size'], 3):
            raise ValueError('Tile cache shape differs from metadata')
        if count > len(tiles):
            raise ValueError('Too few cached tiles')
        return tiles[:count]
    if metadata.get('cache_format') != 'jpeg' or count > metadata['cache_tiles']:
        raise ValueError('Unknown cache format or too few cached tiles')
    size = metadata['tile_size']
    result = np.empty((count, size, size, 3), np.uint8)
    with zipfile.ZipFile(folder/'tiles.zip') as archive:
        if len(archive.namelist()) != metadata['cache_tiles']:
            raise ValueError('Incomplete JPEG tile archive')
        for i in range(count):
            entry = archive.getinfo(f'{i:04d}.jpg')
            if entry.file_size > size*size*6 + 65536:
                raise ValueError('Unexpectedly large encoded tile')
            with archive.open(entry) as stream, Image.open(stream) as image:
                if image.size != (size, size):
                    raise ValueError('JPEG tile dimensions differ from metadata')
                result[i] = np.asarray(image.convert('RGB'))
    return result


def verify_cache(folder, config):
    """Verify decoded tiles and settings before reusing a cache or removing downloaded sources."""
    metadata = json.loads((Path(folder)/'metadata.json').read_text())
    for key in ('level', 'tile_size', 'cache_tiles', 'offset'):
        if metadata[key] != getattr(config, key):
            raise ValueError(f'Cached {key} differs from requested configuration')
    if metadata.get('cache_format', 'npy') != config.cache_format:
        raise ValueError('Cached format differs from configuration')
    if config.cache_format == 'jpeg' and metadata.get('jpeg_quality') != config.jpeg_quality:
        raise ValueError('Cached JPEG quality differs from configuration')
    if config.require_patch_labels and not metadata['has_mask']:
        raise ValueError('Cached label mask is missing')
    read_tiles(folder, metadata, config.cache_tiles)
    return metadata
