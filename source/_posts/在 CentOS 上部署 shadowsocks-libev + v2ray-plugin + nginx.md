---
title: 在 CentOS 上部署 shadowsocks-libev + v2ray-plugin + nginx + Hexo
date: 2020-02-21 12:25:00
tags: linux
toc: false
hidden: true
---

## 1. VPS购买与内核升级、开启BBR

推荐购买国外 CN2 GIA 线路的VPS：<https://www.zhujiceping.com/all-vps/cn2-vps/> 因为不走 CN2，丢包真的太高了，之前 Vultr 东京服务器，非高峰丢包30%，高峰丢包50%，网速大打折扣。

系统使用 CentOS 7。

### 更新 Linux 内核

```bash
uname -sr
rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
rpm -Uvh http://www.elrepo.org/elrepo-release-7.0-2.el7.elrepo.noarch.rpm
yum --enablerepo=elrepo-kernel install kernel-ml
reboot
uname -sr
```

如果内核版本没有变化，则需要设置 GRUB 默认的内核版本

```bash
vi /etc/default/grub
```

设置 `GRUB_DEFAULT=0`。意思是 GRUB 初始化页面的第一个内核将作为默认内核。

```bash
grub2-mkconfig -o /boot/grub2/grub.cfg
reboot
uname -r
```

重启之后内核就更新完毕了。

### 使用 BBR 拥塞算法

```bash
vim /etc/sysctl.conf
```

添加以下内容

```conf
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
```

加载系统参数（正常情况下会输出我们之前加入的内容）

```bash
sysctl -p
```

验证bbr是否已经开启:
若

```bash
sysctl net.ipv4.tcp_available_congestion_control
```

返回

```bash
net.ipv4.tcp_available_congestion_control = bbr cubic reno
```

且

```bash
lsmod | grep bbr
```

输出不为空即成功。

## 2. 网络测速

### 丢包和延迟测试

用 <http://ping.pe> 测试全球不同地区（主要是中国）的延迟和丢包情况。

### 网速测试

用 SpeedTest 测试网速。

```bash
wget -O speedtest-cli https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py
chmod +x speedtest-cli
./speedtest-cli
```

speedtest会自动选择最好的服务器进行测速。也可以选择指定的服务器测速。在 <http://www.speedtest.net/speedtest-servers-static.php> 中找到你想要测速的服务器节点的 `id` ，比如上海电信是3633，那么测速命令就是

```bash
./speedtest-cli --server 3633
```

## 3. 域名注册与 Cloudflare 配置

将域名的 Name server 选到 Cloudflare，再在 Cloudflare 中把域名解析到 VPS 上。注意在 Cloudflare 中只需要开启 DNS功能（记得把解析你的域名到 VPS 上！），不开启CDN功能。（当你的 VPS 被屏蔽时，打开CDN即可恢复，但连接速度可能会下降）。

在SSL选项中选择full。

## 4.安装 Shadowsocks-libev

官方的安装方法有一些[问题](https://github.com/shadowsocks/shadowsocks-libev/issues/2491)浪费了我好长时间，还是用秋大的一键安装脚本吧。

```bash
wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-libev.sh
chmod +x shadowsocks-libev.sh
./shadowsocks-libev.sh
```

卸载方法：
使用 root 用户登录，运行以下命令： `./shadowsocks-libev.sh uninstall`

使用命令：
启动：`/etc/init.d/shadowsocks start`
停止：`/etc/init.d/shadowsocks stop`
重启：`/etc/init.d/shadowsocks restart`
查看状态：`/etc/init.d/shadowsocks status`

需要注意的是，这个命令并不能停止 `v2ray-plugin` 进程，导致重启命令失效，这时候需要手动停止 `v2ray-plugin` 进程。先找到进程的`pid`

```bash
netstat -tulnp|grep ss端口
```

输出：

```bash
tcp6       0      0 :::ss端口              :::*                    LISTEN      1302/v2ray-plugin
```

列出所有占用 yourport 端口的进程。同时显示它的 `pid`（比如这里是1302）。手动杀掉这个进程：

```bash
kill 1302
```

即可正常执行重启命令。

### 5. 安装 GOlang （因为 v2ray-plugin 是用 GO 语言写的）

从 <https://golang.org/dl/> 找到Go语言最新版本的下载地址

```bash
wget https://dl.google.com/go/go1.13.8.linux-amd64.tar.gz
tar -xvf go1.13.8.linux-amd64.tar.gz
sudo mv go /usr/local
```

设置环境变量：

```bash
export GOROOT=/usr/local/go
export GOPATH=$HOME/work
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
```

检查安装：

```bash
go version
```

## 6. 安装 v2ray-plugin

先去 GitHub 上下载最新的 release <https://github.com/shadowsocks/v2ray-plugin/releases>

```bash
wget https://github.com/shadowsocks/v2ray-plugin/releases/download/v1.3.0/v2ray-plugin-linux-amd64-v1.3.0.tar.gz
tar -xzf v2ray-plugin-linux-amd64-v1.3.0.tar.gz
mv v2ray-plugin_linux_amd64 /usr/bin/v2ray-plugin
```

## 6. 申请 SSL 证书

这一步之所以放到后面，是因为 DNS解析需要一段时间。现在 Cloudflare 的 DNS 应该解析完成了。先到 cloudflare 页面，点击右上角的头像，然后点击 My Profile ，在个人信息页面下拉到最后有个 API KEYs -> **Global API KEY**, 复制你的 API KEY, 同时复制你的 CloundFlare 的**登陆邮箱** ， key 和 email 用于申请证书。

```bash
export CF_Email="CloundFlare邮箱"
export CF_Key="API KEY"
curl https://get.acme.sh | sh
~/.acme.sh/acme.sh --issue --dns dns_cf -d 你的域名
```

输出：

```bash
Your cert is in  /root/.acme.sh/你的域名/你的域名.cer
Your cert key is in  /root/.acme.sh/你的域名/你的域名.key
The intermediate CA cert is in  /root/.acme.sh/你的域名/ca.cer
And the full chain certs is there:  /root/.acme.sh/你的域名/fullchain.cer
```

## 7. 配置 Shadowsocks 和 v2ray-plugin

首先新建shadowsocks配置文件的目录，然后创建config.json。

```bash
mkdir /etc/shadowsocks-libev
vim /etc/shadowsocks-libev/config.json
```

配置内容如下：

```json
{
  "server": "127.0.0.1",
  "nameserver": "8.8.8.8",
  "server_port": ss端口,
  "password": "你的密码",
  "method": "chacha20-ietf-poly1305",
  "timeout": 600,
  "no_delay": true,
  "mode": "tcp_and_udp",
  "plugin": "v2ray-plugin",
  "plugin_opts": "server;path=/yourpath;fast-open;host=xxxxxxxxx.com;cert=/证书目录/fullchain.cer;key=/证书目录/xxxxxxxxx.com.key;loglevel=none"
}
```

要注意的几点：

1. server 是 127.0.0.1，即只接受本机流量，让 ss 彻底对外不可见。在后边要使用 nginx 反向代理。
2. `plugin_opts` 中不填写 `tls` ,也因为 nginx 反向代理后转发的是 http 流量。
3. 其中cert和key可不填写，v2ray-plugin会自动搜索acme创建的证书。
4. `yourpath` 是自定义的，在 Nginx 配置中会用到。

## 8. 安装并配置 Nginx

### 下载安装

使用lnmp一键安装脚本，这样安装出来的 Nginx 和官方源的相比，支持更先进的功能（如 `tls1.3`）

```bash
wget http://soft.vpser.net/lnmp/lnmp1.6.tar.gz -cO lnmp1.6.tar.gz && tar zxf lnmp1.6.tar.gz && cd lnmp1.6 && ./install.sh nginx
```

打开之后，可以查看一下防火墙打开的所有的服务

```bash
sudo firewall-cmd --list-service
ssh dhcpv6-client http
```

可以看到，系统已经打开了 http 服务。

反向代理:

需要指出的是 CentOS 7 的 SELinux，使用反向代理需要打开网络访问权限。

```bash
sudo setsebool -P httpd_can_network_connect on
```

打开网络权限之后，反向代理可以使用了。

### 配置 Nginx

创建自己的配置文件：

```bash
vim /usr/local/nginx/conf/vhost/你的域名.conf
```

在其中添加

```json
server {
        listen       443 ssl http2;
        listen       [::]:443 ssl http2;
        server_name  你的域名;     # Your domain.
        root         /home/wwwroot/default;
        ssl_certificate "/root/.acme.sh/你的域名/fullchain.cer";     # Path to certificate
        ssl_certificate_key "/root/.acme.sh/你的域名/你的域名.key";     # Path to private key
        ssl_session_cache shared:SSL:1m;
        ssl_session_timeout  10m;
        ssl_ciphers 'ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES128-SHA256:ECDHE-RSA-AES128-SHA256:ECDHE-ECDSA-AES128-SHA:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA:ECDHE-ECDSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA:ECDHE-RSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA:DHE-RSA-AES256-SHA256:DHE-RSA-AES256-SHA:ECDHE-ECDSA-DES-CBC3-SHA:ECDHE-RSA-DES-CBC3-SHA:EDH-RSA-DES-CBC3-SHA:AES128-GCM-SHA256:AES256-GCM-SHA384:AES128-SHA256:AES256-SHA256:AES128-SHA:AES256-SHA:DES-CBC3-SHA:!DSS';
        ssl_prefer_server_ciphers on;

        location /yourpath { #这里写在ss配置里的yourpath
                proxy_redirect off;
                proxy_pass http://127.0.0.1:yourport; #把流量转发为http到 v2ray-plugin
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection "upgrade";
                proxy_set_header Host $http_host;
        }

                location = / {
                root  /home/wwwroot/default;
                index index.html;
        }
}

}

```

可以先测试以下配置文件是否正确：

```bash
nginx -t
```

配置正确，重载 nginx 设置：

```bash
lnmp nginx reload
```

以上。

-------------------

## references

1. <https://www.vpsserver.com/community/tutorials/3943/how-to-run-speedtest-in-the-terminal-and-choose-speedtest-location-for-what-you-want/>
2. <https://github.com/iMeiji/shadowsocks_install/wiki/shadowsocks-libev-%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85>
3. <https://gist.github.com/Shuanghua/c9c448f9bd12ebbfd720b34f4e1dd5c6>
4. <https://www.coldawn.com/category/linux/>
5. <https://www.wenboz.com/p/af93.html>
6. <https://www.jianshu.com/p/05810dbe129b>