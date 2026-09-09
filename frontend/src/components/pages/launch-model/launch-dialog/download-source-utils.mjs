export function shouldApplyPreferredDownloadSource(
  preferredSource,
  options,
  userChangedDownloadHub
) {
  return (
    !userChangedDownloadHub && options.some((option) => option.value === preferredSource)
  );
}
