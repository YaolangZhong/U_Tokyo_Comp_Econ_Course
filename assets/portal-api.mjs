export const PROGRAMMES = ['master_1', 'master_2', 'phd_1', 'phd_2', 'phd_3'];

export function validateProfile(firstName, lastName, programme) {
  const first = firstName.trim();
  const last = lastName.trim();
  if (!first || !last || first.length > 100 || last.length > 100) {
    throw new Error('Enter your first and second names (up to 100 characters each).');
  }
  if (!PROGRAMMES.includes(programme)) throw new Error('Select your programme and year.');
  return { first_name: first, last_name: last, programme };
}

export function validateConfig(config) {
  if (!config.enabled) return false;
  const url = new URL(config.apiUrl);
  if (url.protocol !== 'https:' || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
    throw new Error('Registration service configuration is invalid.');
  }
  if (!config.publicKey || config.publicKey.startsWith('sb_secret_')) {
    throw new Error('A public client key is required.');
  }
  // Reject a legacy service-role JWT accidentally placed in public configuration.
  if (config.publicKey.split('.').length === 3) {
    let payload;
    try {
      const part = config.publicKey.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
      payload = JSON.parse(atob(part));
    } catch { throw new Error('Invalid public client key.'); }
    if (payload.role !== 'anon') throw new Error('Only an anonymous public client key is allowed.');
  } else if (!config.publicKey.startsWith('sb_publishable_')) {
    throw new Error('Use a publishable public client key.');
  }
  return true;
}
