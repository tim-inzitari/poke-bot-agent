# Public training-dashboard edge

This Caddy edge exposes the existing dashboard through HTTPS while keeping the
dashboard process itself on port `8780`.

- Public clients are challenged with HTTP Basic Authentication.
- RFC1918/link-local LAN clients and Tailscale (`100.64.0.0/10`) bypass the
  challenge.
- Direct LAN/Tailscale access to `http://HOST:8780` is unchanged.
- Port `8780` must never be forwarded by the router. Forward only TCP 80 and
  443 to Bert.
- `/replay-inspector/` opens the separately managed Replay Model Inspector
  through the same HTTPS/access-policy edge. Caddy strips that prefix and
  proxies only to Bert loopback `127.0.0.1:8792`.
- Bert `127.0.0.1:8792` is a managed SSH local forward to the inspector's
  unchanged Elmo loopback listener `127.0.0.1:8791`. Neither inspector port is
  allowed on a LAN/public bind.
- Browser credentials, cookies, origin, referrer, and forwarding headers are
  stripped before the inspector upstream. Only GET is accepted on the
  inspector prefix.

The password is represented only by a bcrypt hash in `Caddyfile`; the plaintext
password is not stored in the repository.

The active GoDaddy subdomain is:

```text
POKEBOT_DASHBOARD_DOMAIN=mc.tsinzitari.com
```

Validate before enabling the service:

```sh
caddy validate \
  --config deploy/caddy/Caddyfile \
  --envfile /path/to/private/dashboard.env
```

The GoDaddy A record must point that subdomain to the router's public IPv4
address. If the ISP uses CGNAT, use a tunnel instead of router port forwarding.

Validate the managed inspector tunnel definition and Caddy route before
activation:

```sh
plutil -lint deploy/launchd/com.pokebot.replay-model-inspector-tunnel.plist
caddy validate --config deploy/caddy/Caddyfile
```

After the tunnel is loaded, `lsof -nP -iTCP:8792 -sTCP:LISTEN` must report only
`127.0.0.1:8792`. A request to `/replay-inspector/api/health` through the HTTPS
edge must succeed, while a non-GET request must return `405` without contacting
the inspector.
