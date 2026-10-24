/**
 * Minimal route used by RunHive's authenticated login-time preloader.
 * RunHiveAuthGate exchanges the one-time ticket and warms the account's
 * backend worker; this page deliberately mounts no data or GPU UI.
 */
export default function RunOSWarmPage() {
  return null
}
