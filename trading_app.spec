# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['complete_trading_app.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'yfinance',
        'pandas',
        'numpy',
        'ta',
        'scikit-learn',
        'matplotlib',
        'tkinter',
        'threading',
        'datetime',
        'requests',
        'googlesearch',
        'textblob',
        'warnings'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


pyt = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyt,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AdvancedStockTrader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)
