import matplotlib.font_manager as fm

# 获取所有字体名称
font_names = [font.name for font in fm.fontManager.ttflist]
with open("font_names.txt", "w") as f:
    for font_name in font_names:
        f.write(font_name + "\n")
print("可用字体列表：", font_names)