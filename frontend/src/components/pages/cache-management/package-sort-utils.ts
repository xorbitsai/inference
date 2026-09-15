import type { VirtualEnvPackage } from '@/types/services';

export type PackageSort = {
  key: 'name' | 'size';
  direction: 'asc' | 'desc';
};

export function sortPackages(
  packages: VirtualEnvPackage[],
  { key, direction }: PackageSort
): VirtualEnvPackage[] {
  const multiplier = direction === 'asc' ? 1 : -1;

  return [...packages].sort((left, right) => {
    const comparison =
      key === 'name'
        ? left.name.localeCompare(right.name, undefined, { sensitivity: 'base' })
        : left.size_bytes - right.size_bytes || left.name.localeCompare(right.name);

    return comparison * multiplier;
  });
}
