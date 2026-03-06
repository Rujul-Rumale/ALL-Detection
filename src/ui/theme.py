"""
Callisto Dark Theme - Medical Dashboard Palette
Designed for high contrast and modern aesthetics on 1080p displays.
"""

THEME = {
    # Base Colors
    "bg": "#1A1B1E",         # Deep Slate (App Background)
    "surface": "#25262B",    # Lighter Slate (Card/Panel Background)
    "sidebar": "#212226",    # Slightly darker surface for nav
    
    # Accents
    "primary": "#4CC9F0",    # Cyan (Action Buttons, Highlights) - Clean, clinical
    "primary_hover": "#3DB5D8",
    
    "accent": "#7209B7",     # Violet (AI Features) - Special magic color
    "accent_hover": "#5F0899",
    
    # Status
    "danger": "#FF4D4D",     # Soft Red (Blast Detection)
    "success": "#40C057",    # Emerald (Healthy Status)
    "warning": "#FFAA00",    # Orange (Processing/Loading)
    
    # Cell Classification
    "blast_border": "#FF4D4D",   # Red border for blast cells
    "healthy_border": "#40C057", # Green border for healthy cells
    "debris_border": "#FFAA00",  # Amber border for debris
    "chart_bg": "#1E1F23",       # Slightly darker for chart areas
    
    # Typography
    "text": "#E9ECEF",       # Off-White (Primary Text)
    "text_dim": "#909296",   # Muted Gray (Secondary Text)
    
    # Fonts
    "font_main": ("Roboto", 14),
    "font_sm": ("Roboto", 12),
    "font_xs": ("Roboto", 10),
    "font_header": ("Roboto", 20, "bold"),
    "font_hero": ("Roboto", 28, "bold"),
    "font_mono": ("JetBrains Mono", 13),
    
    # Geometry
    "corner_radius": 15,
    "border_width": 1,
    "border_color": "#2C2E33"
}
