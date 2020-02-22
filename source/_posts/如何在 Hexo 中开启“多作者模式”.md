---
title: 如何在 Hexo 中开启“多作者模式”
date: 2020-02-22 10:51:00
author: 冬茶
tags: 
    - 冬茶
    - Hexo
---

## 前言

大多数 Hexo 主题都不支持多作者，找到了一种曲径救国的方法。

## 在 Markdown Posts 中的编辑

在每篇 Post 的 Front-matter 中添加字段：

```markdown
author: name
tags: name
```

这样我们把每一篇 Post 都增加了 作者（用来显示作者名称）和标签（用来检索该作者的所有 Post）字段。

## 配置模版文件

因主题各异，以下内容可能需要灵活变通...  

首先在主题目录 `themes/your_theme/layout` 中看一下大概结构。在我的主题中，有：`_partial`, `_pages`， `_plugin` 三个文件夹 和模版 `post.ejs`，`index.ejs`。  

首先打开 `post.ejs`，看到如下字段

```html
<!-- ## Post ## -->
<%- partial('_pages/post') %>
```

打开 `_pages/post.ejs`，看到如下字段

```html
<!-- # Post Header Info # -->
<div class="card-header">
    <%- partial('_partial/post/header-info') %>
</div>
```

提示每篇文章的 Header 内容存在 `_partial/post/header-info` 中，打开并修改：

```diff
<div class="post-header-info">
    <p class="post-header-info-left text-gray">
        # 注释掉头像
-       <img class="author-thumb lazyload" data-src="<%= url_for(theme.img.avatar) %>" src="<%= lazyloadImgSrc %>" alt="<%= config.author %>'s Avatar">
+       <!-- <img class="author-thumb lazyload" data-src="<%= url_for(theme.img.avatar) %>" src="<%= lazyloadImgSrc %>" alt="<%= config.author %>'s Avatar"> -->
        <span><%= date(page.date, config.date_format) %></span>
# 增加作者信息
+       <% if (page.author) {%>
+       <a href=<%= url_for('/tags/' + page.author) %>><%= page.author %></a>
+       <% } %>
# ...
    </p>
    </div>
</div>
```

这样在每篇 Post 中就可以显示作者信息了。

然后还需要在首页中关闭作者的头像，用一样的办法，在 `index.ejs` 中寻找，最后注释掉显示头像的代码行。

## 参考

[基于 Hexo 实现多作者博客](https://bolt.coding.me/blog/2017/03/13/%E5%9F%BA%E4%BA%8E-Hexo-%E5%AE%9E%E7%8E%B0%E5%A4%9A%E4%BD%9C%E8%80%85%E5%8D%9A%E5%AE%A2/)
