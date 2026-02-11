from PIL import Image, ImageDraw
import os

# Icon sizes for different densities
sizes = {
    'mdpi': 48,
    'hdpi': 72,
    'xhdpi': 96,
    'xxhdpi': 144,
    'xxxhdpi': 192
}

base_path = r'c:\Open Source\leukiemea\android_app\app\src\main\res'

for density, size in sizes.items():
    # Create square icon
    img = Image.new('RGB', (size, size), color='#1976D2')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple medical cross
    cross_color = '#FFFFFF'
    thickness = max(size // 12, 4)
    center = size // 2
    arm_length = size // 3
    
    # Horizontal bar
    draw.rectangle(
        [(center - arm_length, center - thickness // 2),
         (center + arm_length, center + thickness // 2)],
        fill=cross_color
    )
    # Vertical bar
    draw.rectangle(
        [(center - thickness // 2, center - arm_length),
         (center + thickness // 2, center + arm_length)],
        fill=cross_color
    )
    
    # Save square icon
    mipmap_dir = os.path.join(base_path, f'mipmap-{density}')
    img.save(os.path.join(mipmap_dir, 'ic_launcher.png'))
    
    # Create round icon (same content, just note it's for round)
    img.save(os.path.join(mipmap_dir, 'ic_launcher_round.png'))

print("Launcher icons created successfully!")
