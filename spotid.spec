# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('interface/templates', 'interface/templates'),
        ('leopard_id/weights/tf_efficientnetv2_b2_pretrained.pth', 'leopard_id/weights'),  # Add pretrained backbone weights
        ('interface/static', 'interface/static'),
        ('interface/app.py', 'interface'),
        ('leopard_id/weights/best-model-cosface.pth', 'leopard_id/weights'),
        ('leopard_id/data/minimum_train_data_cropped', 'leopard_id/data/minimum_train_data_cropped'),
        ('leopard_id/data/histogram_matching', 'leopard_id/data/histogram_matching'),
        ('leopard_id/config_inference.json', 'leopard_id'),
        ('leopard_id/model', 'leopard_id/model'),
        ('leopard_id/losses', 'leopard_id/losses'),
        ('leopard_id/scripts_preprocessing', 'leopard_id/scripts_preprocessing'),
        ('leopard_id/inference_embeddings.py', 'leopard_id'),
    ],
    hiddenimports=[
        'engineio.async_drivers.threading',
        'leopard_id',
        'leopard_id.model',
        'leopard_id.model.EmbeddingNetwork',
        'leopard_id.losses',
        'leopard_id.losses.Cosface',
        'leopard_id.losses.TripletLoss',
        'leopard_id.inference_embeddings',
        'leopard_id.scripts_preprocessing',
        'leopard_id.scripts_preprocessing.background_removal',
        'leopard_id.scripts_preprocessing.bbox_creation',
        'leopard_id.scripts_preprocessing.edge_detection',
        'leopard_id.scripts_preprocessing.run_all_preprocessing',
        'PIL',
        'PIL.Image',
        'numpy',
        'torch',
        'torchvision',
        'torchvision.transforms',
        'waitress',
        'flask',
        'networkx',
    ],
    hookpath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='spotid',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='interface/static/images/logo.jpg'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='spotid'
)
