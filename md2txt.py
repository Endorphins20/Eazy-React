import os

def convert_md_to_txt(folder_path):
    """
    将指定文件夹中的所有 .md 文件转换为 .txt 文件。
    :param folder_path: 包含 .md 文件的文件夹路径
    """
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)

    # 筛选出所有 .md 文件
    md_files = [file for file in files if file.endswith('.md')]

    # 遍历每个 .md 文件并转换为 .txt 文件
    for md_file in md_files:
        md_file_path = os.path.join(folder_path, md_file)  # 完整的 .md 文件路径
        txt_file_path = os.path.join(folder_path, md_file[:-3] + '.txt')  # 对应的 .txt 文件路径

        # 读取 .md 文件内容
        with open(md_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 将内容写入 .txt 文件
        with open(txt_file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Converted {md_file} to {os.path.basename(txt_file_path)}")

# 示例用法
if __name__ == "__main__":
    # 设置目标文件夹路径
    folder_path = r'D:\Master\llm\database\kag\测试数据集'  # 替换为你的文件夹路径

    # 调用函数进行转换
    convert_md_to_txt(folder_path)