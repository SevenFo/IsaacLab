import os
import argparse
import glob


def collect_py_files(paths, output_file):
    """
    收集多个路径中的.py文件内容并写入输出文件
    """
    # 收集所有.py文件路径
    all_files = []
    for path in paths:
        # 处理目录
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".py"):
                        all_files.append(os.path.join(root, file))
        # 处理文件通配符
        elif "*" in path:
            all_files.extend(glob.glob(path, recursive=True))
        # 处理单个文件
        elif os.path.isfile(path) and path.endswith(".py"):
            all_files.append(path)

    # 去重并排序
    all_files = sorted(set(all_files))

    if not all_files:
        print("警告: 没有找到任何.py文件!")
        return

    with open(output_file, "w", encoding="utf-8") as out_f:
        print(f"正在合并 {len(all_files)} 个文件...")

        for file_path in all_files:
            try:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    # 写入文件路径标题
                    out_f.write(f"\n\n{'=' * 80}\n")
                    out_f.write(f"# File: {os.path.abspath(file_path)}\n")
                    out_f.write(f"{'=' * 80}\n\n")

                    # 写入文件内容
                    out_f.write(in_f.read())
                    print(f"已添加: {file_path}")

            except Exception as e:
                print(f"无法读取文件 {file_path}: {str(e)}")

    print(f"\n已完成! 共合并 {len(all_files)} 个文件到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="合并多个路径中的Python文件到单个文本文件",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="要处理的目录/文件路径(支持通配符)")
    parser.add_argument("-o", "--output", required=True, help="输出文件路径")

    args = parser.parse_args()

    collect_py_files(args.paths, args.output)
