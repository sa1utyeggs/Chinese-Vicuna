## Features

- Local HTTP/HTTPS/SOCKS server with authentication support
- Shadowsocks(R), VMess, Trojan, Snell, SOCKS5, HTTP(S) outbound support
- Built-in [fake-ip](https://www.rfc-editor.org/rfc/rfc3089) DNS server that aims to minimize DNS pollution attack impact. DoH/DoT upstream supported.
- Rules based off domains, GEOIP, IP-CIDR or process names to route packets to different destinations
- Proxy groups allow users to implement powerful rules. Supports automatic fallback, load balancing or auto select proxy based off latency
- Remote providers, allowing users to get proxy lists remotely instead of hardcoding in config
- Transparent proxy: Redirect TCP and TProxy TCP/UDP with automatic route table/rule management
- Hot-reload via the comprehensive HTTP RESTful API controller

## Premium

Premium core is proprietary. You can find their release notes and pre-built binaries [here](https://github.com/Dreamacro/clash/releases/tag/premium).

- gvisor/system stack TUN device on macOS, Linux and Windows ([ref](https://github.com/Dreamacro/clash/wiki/Clash-Premium-Features#tun-device))
- Policy routing with [Scripts](https://github.com/Dreamacro/clash/wiki/Clash-Premium-Features#script)
- Load your rules with [Rule Providers](https://github.com/Dreamacro/clash/wiki/Clash-Premium-Features#rule-providers)
- Monitor Clash usage with a built-in profiling engine. ([Dreamacro/clash-tracing](https://github.com/Dreamacro/clash-tracing))

## Getting Started
Documentations are available at [GitHub Wiki](https://github.com/Dreamacro/clash/wiki).

## Development
If you want to build a Go application that uses Clash as a library, check out the [GitHub Wiki](https://github.com/Dreamacro/clash/wiki/Using-Clash-in-your-Golang-program).

## Credits

* [riobard/go-shadowsocks2](https://github.com/riobard/go-shadowsocks2)
* [v2ray/v2ray-core](https://github.com/v2ray/v2ray-core)
* [WireGuard/wireguard-go](https://github.com/WireGuard/wireguard-go)

## License

This software is released under the GPL-3.0 license.
