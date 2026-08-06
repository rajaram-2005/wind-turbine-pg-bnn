# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.0   | ✅ Current         |

## Reporting a Vulnerability

We take the security of Aerovigil PG-BNN seriously. If you discover a security vulnerability, please report it privately.

### Preferred Reporting Method

**Email:** contact@aerovigil.ai

Please include the following in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

*   **Acknowledgment:** Within 48 hours
*   **Initial Assessment:** Within 5 business days
*   **Resolution Timeline:** Depends on severity and complexity

### Security Best Practices

When using this software:

1. **Keep Dependencies Updated**
   ```bash
   pip list --outdated
   pip install --upgrade
   ```

2. **Use Secure Configurations**
   - Never commit secrets or API keys
   - Use environment variables for sensitive data
   - Review `docker/secret.yaml` before deployment

3. **Container Security**
   - Run containers as non-root user
   - Use minimal base images
   - Regularly scan for vulnerabilities

4. **Model Security**
   - Verify model file integrity before loading
   - Use `weights_only=True` in `torch.load()` when possible
   - Keep model files in secure storage

### Disclosure Policy

When we receive a security report, we will:

1. Confirm receipt of the report
2. Assess the severity and impact
3. Develop and test a fix
4. Release the fix
5. Publish a security advisory (if warranted)

We request that you:
- Give us reasonable time to fix the issue
- Do not disclose the vulnerability publicly until we've released a fix
- Do not exploit the vulnerability beyond what's necessary to demonstrate it

## Security Contacts

- **Primary:** contact@aerovigil.ai
- **Alternative:** rajaram@aerovigil.ai

For critical incidents affecting production deployments, please include "URGENT" in the subject line.
