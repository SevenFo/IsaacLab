# Git 修改历史中的错误作者信息

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


# Git 选择应用其他分支的 commit

**方法一：使用 `git cherry-pick` (推荐用于选择性地应用少量 commit)**

`git cherry-pick` 命令允许你选择一个或多个来自其他分支的 commit，并将它们作为新的 commit 应用到你当前所在的分支上。

1.  **确保你的本地仓库有最新的信息（如果其他分支是远程分支）：**
    ```bash
    git fetch origin # 或者其他远程仓库名
    ```
    如果你要 cherry-pick 的分支是本地的，则不需要这一步。

2.  **切换到你的目标分支（你想要将 commit 应用到的分支）：**
    ```bash
    git checkout my-branch # 替换为你的分支名
    ```

3.  **找到你要 cherry-pick 的 commit 的 SHA-1 哈希值：**
    你可以使用 `git log` 查看其他分支的提交历史。
    ```bash
    git log other-branch-name # 替换为包含你想要commit的分支名
    # 例如：git log feature-xyz
    ```
    从输出中复制你想要的 commit 的 SHA-1 哈希值 (通常是一长串字母和数字，比如 `a1b2c3d4e5f6...`)。

4.  **执行 `cherry-pick`：**
    ```bash
    git cherry-pick <commit-sha-1>
    ```
    例如：
    ```bash
    git cherry-pick a1b2c3d4
    ```
    这会在 `my-branch` 上创建一个新的 commit，这个新 commit 的内容（文件更改）与 `a1b2c3d4` 完全相同，但它会有新的 commit 时间戳，并且它的父 commit 是 `my-branch` 当前的 `HEAD`。它的 SHA-1 哈希值也会是全新的。

    **如果要 cherry-pick 多个 commit：**
    你可以按顺序提供多个 SHA-1 值：
    ```bash
    git cherry-pick <commit-sha-1> <commit-sha-2> <commit-sha-3>
    ```
    Git 会按照你给出的顺序（通常是较旧的 commit 在前）逐个应用它们。

    **如果要 cherry-pick 一个连续范围的 commit：**
    假设你要从 `other-branch-name` 上 cherry-pick 从 `commit-A` (不包括) 到 `commit-B` (包括) 之间的所有 commit：
    ```bash
    git cherry-pick <commit-A-sha>^..<commit-B-sha>
    # 例如: git cherry-pick abcdef1^..uvwxyz7
    ```
    注意 `^` 符号，它表示 `commit-A` 的父 commit，所以 `commit-A` 本身也会被包含。
    或者，如果你想包含 `commit-A`：
    ```bash
    git cherry-pick <commit-A-sha>..<commit-B-sha> # 这种写法 commit-A 不会被包含
    # 更清晰的范围：
    # 假设你要应用 commit X, Y, Z，它们在 other-branch 上是连续的
    # 找到 X 的父 commit P
    # git cherry-pick P..Z  (应用 X, Y, Z)
    # 或者简单地:
    # git cherry-pick X Y Z
    ```

5.  **解决冲突（如果发生）：**
    如果 cherry-pick 的 commit 修改了与你当前分支上已有更改相冲突的文件部分，Git 会暂停并提示你解决冲突。
    *   使用 `git status` 查看冲突文件。
    *   手动编辑这些文件以解决冲突。
    *   使用 `git add <resolved-file-name>` 将解决后的文件标记为已解决。
    *   然后继续 cherry-pick 过程：
        ```bash
        git cherry-pick --continue
        ```
    *   如果想放弃当前的 cherry-pick 操作：
        ```bash
        git cherry-pick --abort
        ```

**方法二：使用 `git rebase --onto` (更高级，适用于移动一系列 commit)**

如果你想将一个分支上的一系列 commit（不是整个分支）移动到另一个分支的顶部，`rebase --onto` 非常有用。这和 cherry-pick 连续范围的 commit 效果类似，但有时语义上更清晰。

假设你有如下历史：

```
      A---B---C topic
     /
D---E---F---G main
```

你只想把 `topic` 分支上的 `B` 和 `C` 提交应用到 `main` 分支的 `G` 之后，而不是整个 `topic` 分支。

```bash
# 1. 切换到你希望这些commit最终基于的分支（或者只是一个临时指针）
# git checkout main (如果你想直接在 main 上操作)

# 2. 执行 rebase --onto
# git rebase --onto <新基底> <旧基底> <要移动的分支/commit>
git rebase --onto main topic~2 topic # topic~2 是 A，topic 是 C
                                     # 表示将从 A 之后到 C (即 B 和 C) 的 commit
                                     # 变基到 main 的顶部
```

执行后，`topic` 分支会变成：

```
      A topic~2 (旧的 topic 分支指针可能还在，或者被移动)
     /
D---E---F---G---B'---C' main, topic (新的 topic 指向 C')
```

如果你只是想把这些 commit 应用到当前分支，而不是移动 `topic` 分支本身，可以先切换到目标分支：

```bash
git checkout main
git rebase --onto HEAD topic~2 topic
```

这会把 `B` 和 `C` 变成 `B'` 和 `C'` 应用到 `main` 的顶部。

**方法三：使用 `git format-patch` 和 `git am` (生成补丁文件)**

这是一种更通用的方法，不直接依赖于 Git 的分支结构，而是通过文件。

1.  **在源分支上生成补丁文件：**
    ```bash
    git checkout other-branch-name
    git format-patch -N <commit-sha> # -N 表示从 <commit-sha> 往前的 N 个 commit
    # 或者
    git format-patch <commit-sha-A>^..<commit-sha-B> # 生成 A 到 B (含) 的补丁
    # 例如，只生成一个 commit 的补丁：
    git format-patch -1 <commit-sha>
    ```
    这会生成一个或多个 `.patch` 文件。

2.  **切换到目标分支并应用补丁：**
    ```bash
    git checkout my-branch
    git am <path/to/patch-file.patch> # 应用单个补丁
    # 或者应用目录下所有补丁
    git am <path/to/patches>/*.patch
    ```
    `git am` (apply mail) 会尝试应用这些补丁并创建新的 commit。
