import os
import re


def rename_star_photos_safe(folder_path, preview_mode=True):
    """
    安全批量重命名（带预览模式）

    参数:
        folder_path: 目标文件夹路径
        preview_mode: 预览模式开关。为True时只显示将要做的更改，不实际执行。
    """
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    file_info = []

    for f in all_files:
        # 更灵活地匹配可能的文件名变体
        match = re.match(r'^IMG_\d{8}_\d{6}_(\d+)\.jpg$', f)
        if not match:
            # 如果第一次没匹配上，尝试匹配你之前可能生成的 TIMEBURST{序号}.jpg 格式
            match = re.match(r'^TIMEBURST(\d+)\.jpg$', f)
        if match:
            seq_num = int(match.group(1))
            file_info.append((f, seq_num))

    if not file_info:
        print("未找到符合命名规则（IMG_*_序号.jpg 或 TIMEBURST序号.jpg）的文件。")
        return

    file_info.sort(key=lambda x: x[1])
    max_num = max([info[1] for info in file_info])
    pad_digits = max(3, len(str(max_num)))

    print(f"📁 在文件夹中找到 {len(all_files)} 个jpg文件，其中 {len(file_info)} 个符合命名规则。")
    print(f"🔢 最大序号为 {max_num}，将使用 {pad_digits} 位数字进行补零。")
    print("=" * 60)

    renamed_count = 0
    for old_name, seq_num in file_info:
        prefix = str(seq_num).zfill(pad_digits)
        new_name = f"{prefix}_TIMEBURST{seq_num}.jpg"

        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # 检查新文件名是否可能冲突
        if os.path.exists(new_path) and old_name != new_name:
            print(f"⚠️  警告：跳过 '{old_name}'，因为目标文件 '{new_name}' 已存在。")
            continue

        if preview_mode:
            print(f"[预览] {old_name:35} -> {new_name}")
        else:
            try:
                os.rename(old_path, new_path)
                print(f"✓ 已重命名: {old_name:30} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"✗ 重命名 {old_name} 时出错：{e}")

    print("=" * 60)
    if preview_mode:
        print(f"📋 预览完成。以上是计划对 {len(file_info)} 个文件进行的更改。")
        print("🔧 如需实际执行重命名，请将脚本中的 'preview_mode=True' 改为 'preview_mode=False' 并重新运行。")
    else:
        print(f"✅ 实际操作完成！成功重命名 {renamed_count} 个文件。")


# ================ 主程序开始 ================
if __name__ == "__main__":
    # 配置区
    target_folder = r"E:\大疆action5视频\2.13星空\111"  # 你的文件夹路径
    PREVIEW_MODE = False  # True = 仅预览， False = 实际执行

    # 检查路径
    if not os.path.exists(target_folder):
        print(f"❌ 错误：文件夹路径不存在 - {target_folder}")
    else:
        print("星空照片批量重命名工具")
        print(f"目标文件夹：{target_folder}")
        print(f"模式：{'预览模式（安全）' if PREVIEW_MODE else '执行模式'}")
        print("-" * 40)

        # 询问确认（如果在执行模式）
        if not PREVIEW_MODE:
            confirm = input(f"⚠️  即将实际重命名文件，此操作不可撤销！确定要继续吗？(输入 yes 确认): ")
            if confirm.lower() != 'yes':
                print("操作已取消。")
                exit()

        # 执行函数
        rename_star_photos_safe(target_folder, preview_mode=PREVIEW_MODE)

        # 最终提示
        if not PREVIEW_MODE:
            print("\n💡 提示：现在你的文件命名类似 '001_TIMEBURST1.jpg'。")
            print("在StarStax或其他任何软件中，只需选择『按文件名排序』即可获得完美顺序。")