**推荐方法：使用 `git rebase -i` (交互式变基)**

这是针对修改少量特定 commit 的推荐方法。

1.  **确认你的 Git 配置是正确的（防止以后再出错）：**
    ```bash
    git config --global user.name "你的正确名字"
    git config --global user.email "你的正确邮箱@example.com"
    # 如果只想为当前仓库设置，去掉 --global
    # git config user.name "你的正确名字"
    # git config user.email "你的正确邮箱@example.com"
    ```

2.  **找出需要修改的 commit：**
    你需要找到这两个错误 commit *之前* 的那个 commit 的 SHA-1 哈希值。
    使用 `git log` 查看提交历史。假设你要修改的是最近的两个 commit。

    ```bash
    git log
    ```
    记下你要修改的 commit 的父 commit 的哈希值，或者简单地使用 `HEAD~N`，其中 N 是你要往回看的 commit 数量（包括你想修改的）。如果你想修改最近的两个 commit，你需要 rebase `HEAD~2` 之前的 commit。

3.  **开始交互式变基：**
    假设你要修改的两个 commit 是最近的两个，那么命令是：
    ```bash
    git rebase -i HEAD~2
    ```
    如果你知道那两个错误 commit 之前的 commit 的哈希值（比如 `abcdef123`），则：
    ```bash
    git rebase -i abcdef123
    ```
    这会打开一个文本编辑器，列出从 `HEAD~2` (或 `abcdef123`) *之后* 到当前 `HEAD` 的所有 commit。最早的 commit 在最上面。

4.  **标记要修改的 commit：**
    在打开的编辑器中，你会看到类似这样的内容：
    ```
    pick <commit_hash_1> <commit_message_1>
    pick <commit_hash_2> <commit_message_2>
    # ...
    ```
    找到那两个作者信息错误的 commit，将它们前面的 `pick` 改为 `edit` (或者简写 `e`)。
    例如：
    ```
    edit <commit_hash_bad_1> <commit_message_bad_1>
    edit <commit_hash_bad_2> <commit_message_bad_2>
    ```
    保存并关闭编辑器。

5.  **逐个修改 commit 作者信息：**
    Git 会自动停在第一个标记为 `edit` 的 commit 上。
    此时，你可以使用 `git commit --amend` 来修改作者信息：
    ```bash
    git commit --amend --author="你的正确名字 <你的正确邮箱@example.com>" --no-edit
    ```
    `--no-edit` 参数表示你不想修改 commit message，只想修改作者。如果你也想修改 message，去掉这个参数。

    修改完成后，让 rebase 继续：
    ```bash
    git rebase --continue
    ```
    Git 会接着停在下一个标记为 `edit` 的 commit 上。重复上面的 `git commit --amend` 和 `git rebase --continue` 步骤，直到所有标记为 `edit` 的 commit 都被处理完毕。

6.  **强制推送到远程仓库：**
    **再次警告：这一步会重写远程历史！确保你的协作者知道这件事！**
    ```bash
    git push --force-with-lease <远程仓库名> <分支名>
    # 例如:
    # git push --force-with-lease origin main
    ```
    `--force-with-lease` 比 `--force` 更安全一点，它会检查远程分支在你上次 `fetch` 之后是否被其他人更新过，如果是，则推送失败，防止覆盖他人的新工作。如果只有你自己在这个分支上工作，或者你非常确定，也可以用 `--force`。

**另一种方法：使用 `git filter-repo` (更强大，适用于更复杂或大量的修改)**

`git filter-branch` 曾经是做这类事情的工具，但它复杂且慢，现在官方推荐使用 `git-filter-repo`。你需要先安装它。

1.  **安装 `git-filter-repo`：**
    参考其官方文档安装：[https://github.com/newren/git-filter-repo/blob/main/INSTALL.md](https://github.com/newren/git-filter-repo/blob/main/INSTALL.md)
    通常可以通过 pip 安装： `pip install git-filter-repo`

2.  **创建一个 `.mailmap` 文件 (推荐方式)：**
    在你的仓库根目录下创建一个名为 `.mailmap` 的文件，内容格式如下：
    ```
    Correct Name <correct.email@example.com> Wrong Name <wrong.email@example.com>
    ```
    这一行表示，所有由 "Wrong Name <wrong.email@example.com>" 提交的 commit，其作者信息都会被更正为 "Correct Name <correct.email@example.com>"。

3.  **运行 `git filter-repo`：**
    ```bash
    git filter-repo --mailmap .mailmap
    ```
    这个命令会处理整个仓库历史，根据 `.mailmap` 文件更新所有匹配的 commit 的作者和提交者信息。

    如果你不想用 `.mailmap`，也可以直接用回调函数 (更灵活，但可能更复杂)：
    ```bash
    git filter-repo --commit-callback '
      if commit.author_email == b"wrong.email@example.com":
        commit.author_name = b"Correct Name"
        commit.author_email = b"correct.email@example.com"
      # 如果提交者(committer)信息也错了，也需要修改
      if commit.committer_email == b"wrong.email@example.com":
        commit.committer_name = b"Correct Name"
        commit.committer_email = b"correct.email@example.com"
    '
    ```
    **注意**: `git-filter-repo` 会自动移除你的远程配置，你需要重新添加：
    ```bash
    git remote add origin <你的远程仓库URL>
    ```

4.  **强制推送到远程仓库：**
    同上一步的 `git push --force-with-lease`。
    ```bash
    git push --force-with-lease origin <分支名>
    ```

**给协作者的建议：**
如果你的协作者已经拉取了旧的历史记录，他们需要进行以下操作来同步：
1.  确保他们本地没有未提交的更改（可以 `git stash`）。
2.  获取最新的远程历史：`git fetch origin`
3.  如果他们在错误历史的分支上没有自己的新提交，可以重置到新的远程分支：
    ```bash
    git checkout <分支名>
    git reset --hard origin/<分支名>
    ```
4.  如果他们在错误历史的分支上有自己的新提交，他们需要将自己的提交变基 (rebase) 到新的远程分支上：
    ```bash
    git checkout <他们的特性分支>
    git rebase origin/<主分支名> # 假设主分支历史被修改了
    # 或者如果他们直接在被修改的分支上工作
    # git rebase origin/<被修改的分支名>
    ```
    这期间可能会有冲突，需要解决。
