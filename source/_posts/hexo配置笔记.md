---
title: Hexo配置笔记
date: 2020-2-21 12:55:00
tags: 
  - Hexo
  - 冬茶
author: 冬茶
---

## 1. 安装 Hexo 博客

以下内容需要 git 基础知识。

### GitHub 创建个人仓库

登陆 GitHub，创建新仓库，仓库名为：用户名.github.io

### 在本地安装 git 和 Node.js

首先在你的本地计算机上安装 git 和 Node.js，过程省略。

git 安装完毕后，设置user.name和user.email配置信息：

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub注册邮箱"
```

生成ssh密钥文件：

```bash
ssh-keygen -t rsa -C "你的GitHub注册邮箱"
```

然后直接三个回车即可，默认不需要设置密码  
然后找到生成的.ssh的文件夹中的id_rsa.pub密钥，将内容全部复制。  
打开GitHub_Settings_keys 页面，新建new SSH Key。  
Title为标题，任意填即可，将刚刚复制的id_rsa.pub内容粘贴进去，最后点击Add SSH key。  
在Git Bash中检测GitHub公钥设置是否成功，输入 ssh git@github.com 检测。

### 在本地安装 Hexo

使用npm命令安装Hexo，输入：

```bash
npm install -g hexo-cli
```

安装完毕后进入一个存放博客的文件夹。输入：

```bash
hexo init blog
```

则会在当前目录创建一个 blog 文件夹。进入这个文件夹后输入

```bash
hexo g \\to generate
hexo s \\to start a server
```

完成后，打开浏览器输入地址：<localhost:4000> 即可看到你的 Hexo 博客。

### 在服务器配置git

首先在服务器上安装好 git，并配置好全局 git 的用户名和邮箱，如下：

```bash
git config --global user.name "your name"
git config --global user.email "your email"
```

创建 git 用户：

```bash
adduser git
```

新建相关目录和权限：

```bash
mkdir /home/git
cd /home/git
git init --bare hexo.git
chown -R git:git hexo.git
chown -R git:git /home/wwwroot/default
```

获取本地机器 SSH 的公钥：

```bash
cat ~/.ssh/id_rsa.pub
```

复制公钥，将公钥写入服务器的 /home/git/.ssh/authorized_keys 文件中。  
在 /home/git/hexo.git/hooks/post-receive 文件中写入：

```sh
#!/bin/sh
git --work-tree=/home/wwwroot/default --git-dir=/home/git/hexo.git checkout -f
```

### 完成本地部署配置

修改本地机器上的 _config.yml 文件，在 deploy 块中添加服务器上刚创建的 hexo.git 仓库和 GitHub 仓库：

```yml
deploy:
  type: git
  repo:
        github: git@github.com:用户名/仓库名.git,master
        hexo: git@域名:/home/git/hexo.git,master
```

之后每次发布博客时，只需要在本地执行

```bash
hexo d
```

即可完成在 GitHub 和 服务器上的自动部署。

将_config.yml中的post_asset_folder选项置为true

这样可以方便地为每篇博文配图。

### 增加一个live2d

参考[一个看板娘入住你的个人博客只需要三步](https://lexburner.github.io/live2d/)

## 2. 使用 GitHub 完成多终端同步 Hexo

首先在博客根目录下新建 `.gitignore` 文件，我的内容如下：

```txt
.DS_Store
Thumbs.db
db.json
*.log
node_modules/
public/
.deploy*/
```

在博客根目录下配置仓库：

```bash
git init  //初始化本地仓库
git add -A
git commit -m "Blog Source Hexo"
git branch hexo  //新建hexo分支
git checkout hexo  //切换到hexo分支上
git remote add origin git@github.com:yourname/yourname.github.io.git  //将本地与Github项目对接
git push origin hexo  //push到Github项目的hexo分支上
```

在另一终端完成clone和push更新

```bash
git clone -b hexo git@github.com:yourname/yourname.github.io.git  //将Github中hexo分支clone到本地
cd  yourname.github.io  //切换到刚刚clone的文件夹内
npm install    //注意，这里一定要切换到刚刚clone的文件夹内执行，安装必要的所需组件，不用再init
hexo new post "new blog name"   //新建一个.md文件，并编辑完成自己的博客内容
git add source  //经测试每次只要更新source中的文件到Github中即可，因为只是新建了一篇新博客
git commit -m "XX"
git push origin hexo  //更新分支
hexo d -g   //push更新完分支之后将自己写的博客对接到自己搭的博客网站上，同时同步了Github中的master
```

不同终端间愉快地玩耍

```bash
git pull origin hexo  //先pull完成本地与远端的融合
hexo new post " new blog name"
git add source
git commit -m "XX"
git push origin hexo
hexo d -g
```

-------------------

### references

1. <https://zhuanlan.zhihu.com/p/26625249>
2. <https://zhuanlan.zhihu.com/p/58654392>
3. <https://www.jianshu.com/p/5cf20649f328>
4. <https://blog.csdn.net/Monkey_LZL/article/details/60870891>
5. <https://www.yanbinghu.com/2017/03/31/36088.html>
6. <https://righere.github.io/2016/10/10/install-hexo/>