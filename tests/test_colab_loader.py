"""Exercise the standalone notebook loader without Colab, GPU or network."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import yaml


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/training_reproducibility.ipynb'


def notebook_scope():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    scope = {'Path': Path}
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'def load_archive(' in source:
            exec(compile(source, str(NOTEBOOK), 'exec'), scope)
    return scope


class ColabLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.scope = notebook_scope()

    def archive(self, name='YOLO_Thermal', prefix='wrapper/YOLO_Thermal/'):
        archive = self.base / f'{name}.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(prefix + 'data.yaml', yaml.safe_dump({
                'path': '/Users/old-machine/datasets',
                'train': 'C:/old/images/train', 'val': '/old/images/val',
                'test': '/old/images/test', 'names': {0: 'Deer'},
                'download': 'never execute this archive instruction',
            }))
            for split in ('train', 'val'):
                z.writestr(prefix + f'images/{split}/{split}.jpg', b'image')
                z.writestr(prefix + f'labels/{split}/{split}.txt', '')
                z.writestr(prefix + f'labels/{split}.cache', b'stale training cache')
        return archive

    def test_nested_zip_rewrites_paths_and_preserves_source_and_names(self):
        archive = self.archive()
        source_bytes = archive.read_bytes()
        root = self.scope['prepare_dataset'](archive, 'YOLO_Thermal', self.base / 'extracted')
        config = yaml.safe_load((root / 'data.yaml').read_text())
        self.assertEqual(config['path'], str(root))
        self.assertEqual(config['train'], 'images/train')
        self.assertEqual(config['val'], 'images/val')
        self.assertEqual(config['names'], {0: 'Deer'})
        self.assertNotIn('test', config)
        self.assertNotIn('download', config)
        self.assertEqual(archive.read_bytes(), source_bytes)
        self.assertEqual((root / 'labels/train/train.txt').read_text(), '')
        self.assertFalse((root / 'labels/train.cache').exists())

    def test_completed_extraction_can_be_reused(self):
        archive = self.archive()
        prepare = self.scope['prepare_dataset']
        destination = self.base / 'extracted'
        self.assertEqual(prepare(archive, 'thermal', destination), prepare(archive, 'thermal', destination))

    def test_rejects_path_traversal(self):
        archive = self.base / 'unsafe.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../outside.txt', 'unsafe')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive entry'):
            self.scope['prepare_dataset'](archive, 'thermal', self.base / 'extracted')
        self.assertFalse((self.base / 'outside.txt').exists())

    def test_rejects_raw_video_zip(self):
        archive = self.base / 'video.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('raw.mp4', 'video')
        with self.assertRaisesRegex(ValueError, 'annotated YOLO dataset'):
            self.scope['prepare_dataset'](archive, 'thermal', self.base / 'extracted')

    def test_multiple_versions_require_selection_and_extract_only_selected_version(self):
        archive = self.archive(prefix='YOLO_Thermal/')
        with zipfile.ZipFile(archive, 'a') as z:
            z.writestr('data.yaml', 'names: [Human]\n')
            for split in ('train', 'val'):
                z.writestr(f'images/{split}/other.jpg', b'other')
                z.writestr(f'labels/{split}/other.txt', '')
        destination = self.base / 'extracted'
        prepare = self.scope['prepare_dataset']
        with self.assertRaisesRegex(ValueError, 'multiple dataset versions'):
            prepare(archive, 'YOLO_Thermal', destination)
        self.assertFalse(destination.exists())
        root = prepare(archive, 'YOLO_Thermal', destination, 'YOLO_Thermal')
        self.assertEqual(yaml.safe_load((root / 'data.yaml').read_text())['names'], {0: 'Deer'})
        self.assertFalse((destination / 'images').exists())

    def test_mount_copies_zip_and_reuses_cache(self):
        original = self.archive()
        load = self.scope['load_archive']
        args = ('YOLO_Thermal', 'file-id', 'mount', self.base / 'cache', self.base)
        cached = load(*args)
        self.assertEqual(cached.read_bytes(), original.read_bytes())
        self.assertEqual(load(*args), cached)
        self.assertNotEqual(load('YOLO_Thermal', 'another-id', 'mount', self.base / 'cache', self.base), cached)

    def test_missing_mounted_file_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'DRIVE_DATA_DIR'):
            self.scope['load_archive']('YOLO_RGB', 'id', 'mount', self.base / 'cache', self.base)

    def test_all_python_cells_compile(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(line for line in cell['source'] if not line.startswith('%'))
                compile(source, str(NOTEBOOK), 'exec')


if __name__ == '__main__':
    unittest.main()
