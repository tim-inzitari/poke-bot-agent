# Public training-dashboard edge

This Caddy edge exposes the existing dashboard through HTTPS while keeping the
dashboard process itself on port `8780`.

- Public clients are challenged with HTTP Basic Authentication.
- RFC1918/link-local LAN clients and Tailscale (`100.64.0.0/10`) bypass the
  challenge.
- Direct LAN/Tailscale access to `http://HOST:8780` is unchanged.
- Port `8780` must never be forwarded by the router. Forward only TCP 80 and
  443 to Bert.

The password is represented only by a bcrypt hash in `Caddyfile`; the plaintext
password is not stored in the repository.

Set the actual GoDaddy subdomain in a private environment file:

```text
POKEBOT_DASHBOARD_DOMAIN=app.yourdomain.com
```

Validate before enabling the service:

```sh
caddy validate \
  --config deploy/caddy/Caddyfile \
  --envfile /path/to/private/dashboard.env
```

The GoDaddy A record must point that subdomain to the router's public IPv4
address. If the ISP uses CGNAT, use a tunnel instead of router port forwarding.
