'use client';

import { type MouseEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  ExternalLink,
  Info,
  Loader2,
  RefreshCw,
  Rocket,
  Search,
  Star,
  Trash2,
} from 'lucide-react';
import request from '@/lib/request';
import PageContainer from '@/components/ui/page-container';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { RadioGroup } from '@/components/ui/radio-group';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { useI18n } from '@/contexts/i18n-context';
import { cn } from '@/lib/utils';
import { ModelType, CUSTOM_MODEL_OPTIONS } from '@/constants';
import {
  LAUNCH_MODEL_ROUTE_TABS,
  COLLECTION_STORAGE_KEY,
  LAUNCH_MODEL_UPDATE_OPTIONS,
} from '@/constants/launch';
import LaunchDialog from './launch-dialog/launch-dialog';
import CacheManagementDialog from './cache-management.dialog';
import EnvManagementDialog from './env-management-dialog';
import CustomEditDialog from './custom-edit-dialog';
import type { VirtualEnv } from '@/types/services';
import type { CatalogModel, RouteModelType, RequestModelType } from './types';
import {
  getString,
  isRecord,
  normalizeModels,
  getLaunchModelEndpointType,
  getSortedModels,
} from './utils';

interface LaunchModelProps {
  routeType: RouteModelType;
  initialCustomType: RequestModelType;
}

const LaunchModel = ({ routeType, initialCustomType }: LaunchModelProps) => {
  const { t } = useI18n();
  const router = useRouter();
  const isCustomRoute = routeType === ModelType.Custom;
  const [gpuAvailable, setGPUAvailable] = useState(-1);
  const [customType, setCustomType] = useState<RequestModelType>(initialCustomType);
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [virtualenvs, setVirtualenvs] = useState<VirtualEnv[]>([]);
  const [loading, setLoading] = useState(false);
  const [updateLoading, setUpdateLoading] = useState(false);
  const [refreshType, setRefreshType] = useState<RequestModelType>(ModelType.LLM);
  const [query, setQuery] = useState('');
  const [ability, setAbility] = useState('');
  const [status, setStatus] = useState('');
  const [favorites, setFavorites] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<CatalogModel>();
  const [deleteModel, setDeleteModel] = useState<CatalogModel>();
  const requestType = isCustomRoute ? customType : (routeType as RequestModelType);
  const showAbilityFilter = ![ModelType.Embedding, ModelType.Rerank, ModelType.Custom].includes(
    routeType
  );
  const showStatusFilter = !isCustomRoute;

  const statusOptions = [
    { label: t('launchModel.cached'), value: 'cached' },
    { label: t('launchModel.favorite'), value: 'favorites' },
  ];
  const customTypesRadioOptions = useMemo(
    () => CUSTOM_MODEL_OPTIONS.map((item) => ({ value: item.value, label: t(item.labelKey) })),
    []
  );
  const fetDevices = useCallback(async () => {
    try {
      const res = await request.get('/v1/cluster/devices');
      setGPUAvailable(parseInt(res, 10));
    } catch {}
  }, []);
  const fetchVirtualenvs = useCallback(async () => {
    const res = await request.get('/v1/virtualenvs');
    setVirtualenvs(Array.isArray(res?.list) ? res.list : []);
  }, []);

  const fetchModels = useCallback(
    async (nextType = requestType) => {
      setLoading(true);

      try {
        const endpointType = getLaunchModelEndpointType(nextType);
        const url = isCustomRoute
          ? `/v1/model_registrations/${endpointType}`
          : `/v1/model_registrations/${endpointType}?detailed=true`;
        const res = await request.get(url);
        const modelList = Array.isArray(res) ? res.filter(isRecord) : [];
        const filteredList = modelList.filter((item) =>
          isCustomRoute ? item.is_builtin === false : item.is_builtin === true
        );
        // Setting `detailed=true` when fetching custom models avoids using Promise.all for details,
        // but it's slow when there are too many LLM models.
        // TODO: Add an `is_builtin` param to the backend `/v1/model_registrations/{modelType}` endpoint later.
        // This will save us from fetching single model details one by one.
        const modelsWithDetails = isCustomRoute
          ? await Promise.all(
              filteredList.map(async (item) => {
                const modelName = getString(item, ['model_name']);

                if (!modelName) {
                  return item;
                }

                try {
                  const detail = await request.get(
                    `/v1/model_registrations/${endpointType}/${encodeURIComponent(modelName)}`
                  );

                  return isRecord(detail)
                    ? {
                        ...detail,
                        ...item,
                        model_name: modelName,
                        is_builtin: false,
                      }
                    : item;
                } catch {
                  return item;
                }
              })
            )
          : filteredList;
        setModels(normalizeModels(modelsWithDetails));
      } catch {
        setModels([]);
      } finally {
        setLoading(false);
      }
    },
    [isCustomRoute, requestType]
  );

  useEffect(() => {
    setAbility('');
    setStatus('');
    setQuery('');

    if (isCustomRoute) {
      setCustomType(initialCustomType);
      setRefreshType(ModelType.LLM);
      return;
    }

    setRefreshType(routeType as RequestModelType);
  }, [isCustomRoute, initialCustomType, routeType]);

  useEffect(() => {
    fetDevices();
  }, [fetDevices]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  useEffect(() => {
    fetchVirtualenvs();
  }, [fetchVirtualenvs]);

  useEffect(() => {
    const rawValue = window.localStorage.getItem(COLLECTION_STORAGE_KEY);

    if (!rawValue) return;

    try {
      const value = JSON.parse(rawValue);

      if (Array.isArray(value)) {
        setFavorites(value.map(String));
      }
    } catch {
      setFavorites([]);
    }
  }, []);

  const abilityOptions = useMemo(() => {
    const seen = new Set<string>();
    const options: Array<{ label: string; value: string }> = [];

    models.forEach((model) => {
      model.abilities.forEach((item) => {
        if (seen.has(item)) return;

        seen.add(item);
        options.push({ label: t(`launchModel.${item}`), value: item });
      });
    });

    return options;
  }, [models, t]);

  const visibleModels = useMemo(() => {
    const keyword = query.trim().toLowerCase();

    const filteredModels = models.filter((model) => {
      const matchesKeyword =
        !keyword ||
        model.model_name.toLowerCase().includes(keyword) ||
        model.model_description.toLowerCase().includes(keyword);
      const matchesAbility = !showAbilityFilter || !ability || model.abilities.includes(ability);
      const matchesStatus =
        !showStatusFilter ||
        !status ||
        (status === 'cached' && model.cached) ||
        (status === 'favorites' && favorites.includes(model.model_name));

      return matchesKeyword && matchesAbility && matchesStatus;
    });

    return getSortedModels(filteredModels, favorites);
  }, [ability, favorites, models, query, showAbilityFilter, showStatusFilter, status]);

  const onTabChange = (value: string) => {
    const target = LAUNCH_MODEL_ROUTE_TABS.find((item) => item.key === value);

    if (target) {
      router.push(`/launch-model/${target.path}`);
    }
  };
  const updateModels = async () => {
    setUpdateLoading(true);
    try {
      await request.post('/v1/models/update_type', { model_type: refreshType.toLowerCase() });
      if (isCustomRoute) {
        fetchModels(refreshType);
      } else {
        onTabChange(refreshType);
      }
    } finally {
      setUpdateLoading(false);
    }
  };
  const handleFavorite = (event: MouseEvent<HTMLButtonElement>, model: CatalogModel) => {
    event.stopPropagation();

    setFavorites((prev) => {
      const next = prev.includes(model.model_name)
        ? prev.filter((item) => item !== model.model_name)
        : [...prev, model.model_name];

      window.localStorage.setItem(COLLECTION_STORAGE_KEY, JSON.stringify(next));

      return next;
    });
  };
  const handleDelete = (event: MouseEvent<HTMLButtonElement>, model: CatalogModel) => {
    event.stopPropagation();
    setDeleteModel(model);
  };
  const handleDeleteModel = () => {
    if(!deleteModel) return;
    request
      .delete(`/v1/model_registrations/${customType}/${deleteModel?.model_name}`)
      .then(() => {
        fetchModels(customType);
      })
      .finally(() => setDeleteModel(undefined));
  };
  const handleDetail = (event: MouseEvent<HTMLButtonElement>, model: CatalogModel) => {
    event.stopPropagation();

    if (model.detailUrl) {
      window.open(model.detailUrl, '_blank', 'noopener,noreferrer');
    }
  };

  const renderModelCard = (model: CatalogModel) => {
    const isFavorite = favorites.includes(model.model_name);
    const hasVirtualenv = virtualenvs.some((item) => item.model_name === model.model_name);
    const tags = [...model.abilities, ...model.languages];
    return (
      <div
        key={model.model_name}
        role="button"
        tabIndex={0}
        className={cn(
          'group flex min-h-[240px] cursor-pointer flex-col rounded-lg border border-border bg-card p-5 text-left transition-all duration-200',
          'hover:-translate-y-1 hover:border-primary/50 hover:shadow-lg hover:shadow-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
        )}
      >
        <div className="flex items-start justify-between gap-3">
          <h3 className="line-clamp-1 text-lg font-semibold text-foreground truncate">
            {model.model_name}
          </h3>
          {isCustomRoute ? (
            <div className="flex gap-1 shrink-0">
              <CustomEditDialog model={model} modelType={customType} />
              <button
                type="button"
                aria-label="Delete model"
                onClick={(event) => handleDelete(event, model)}
                className="rounded-full p-1 text-muted-foreground transition-colors hover:text-destructive"
              >
                <Trash2 className="size-5" />
              </button>
            </div>
          ) : (
            <button
              type="button"
              aria-label="Favorite model"
              onClick={(event) => handleFavorite(event, model)}
              className="shrink-0 rounded-full p-1 text-muted-foreground transition-colors hover:text-primary"
            >
              <Star className={cn('size-5', isFavorite && 'fill-primary text-primary')} />
            </button>
          )}
        </div>

        <div className="mt-4 flex flex-wrap gap-1.5 ">
          {tags.map((tag) => (
            <span
              key={tag}
              className="shrink-0 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-xs font-medium text-muted-foreground"
            >
              {tag}
            </span>
          ))}
          {model.cached && (
            <CacheManagementDialog modelDetail={model} onCacheDelete={fetchModels} />
          )}
          {hasVirtualenv && (
            <EnvManagementDialog modelDetail={model} onEnvDelete={fetchVirtualenvs} />
          )}
        </div>

        <div className="mt-4 flex-1">
          <p className="line-clamp-3 text-sm leading-6 text-muted-foreground">
            {model.model_description}
          </p>
        </div>

        <div
          className={cn(
            'mt-5 flex items-center gap-3',
            isCustomRoute ? 'justify-end' : 'justify-between'
          )}
        >
          {!isCustomRoute && (
            <button
              type="button"
              onClick={(event) => handleDetail(event, model)}
              className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
            >
              <ExternalLink className="size-3.5" />
              Details
            </button>
          )}
          <Button
            size="sm"
            onClick={(event) => {
              event.stopPropagation();
              setSelectedModel(model);
            }}
          >
            <Rocket />
            Launch
          </Button>
        </div>
      </div>
    );
  };

  return (
    <PageContainer title={t('menu.launchModel')}>
      <Tabs value={routeType} onValueChange={onTabChange} className="w-full gap-6">
        <div className="flex items-end justify-between gap-4 border-b border-border/80">
          <TabsList className="min-w-0 flex-1 justify-start bg-transparent p-0 h-auto rounded-none overflow-x-auto">
            <div className="flex space-x-4">
              {LAUNCH_MODEL_ROUTE_TABS.map((item) => (
                <TabsTrigger
                  value={item.key}
                  key={item.key}
                  className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
                >
                  {t(item.labelKey)}
                </TabsTrigger>
              ))}
            </div>
          </TabsList>
          <div className="relative z-20 mb-1 flex shrink-0">
            <Select
              value={refreshType}
              onChange={(value) => setRefreshType(value as RequestModelType)}
              options={LAUNCH_MODEL_UPDATE_OPTIONS}
              allowClear={false}
              className="w-32 rounded-r-none"
            />
            <Button onClick={updateModels} disabled={updateLoading} className="rounded-l-none">
              <RefreshCw className={cn(updateLoading ? 'animate-spin' : '')} />
              {t('common.update')}
            </Button>
          </div>
        </div>

        <TabsContent value={routeType} className="space-y-5">
          {isCustomRoute && (
            <div className="border-b border-border/80 pb-4 flex items-center gap-4 pl-4">
              <span className="font-medium">{t('launchModel.modelType')}:</span>
              <RadioGroup
                value={customType}
                onChange={(value) => {
                  setCustomType(value as RequestModelType);
                  setRefreshType(value as RequestModelType);
                  setAbility('');
                  setStatus('');
                  setQuery('');
                }}
                options={customTypesRadioOptions}
                className="flex flex-wrap gap-6"
              />
            </div>
          )}

          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="relative w-full md:max-w-sm">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t('launchModel.search')}
                className="h-10 pl-9"
              />
            </div>
            <div className="flex gap-3">
              {showAbilityFilter && (
                <Select
                  value={ability}
                  onChange={(value) => setAbility(value ?? '')}
                  options={abilityOptions}
                  placeholder={t('launchModel.modelAbility')}
                  allowClear
                  className="w-40"
                />
              )}
              {showStatusFilter && (
                <Select
                  value={status}
                  onChange={(value) => setStatus(value ?? '')}
                  options={statusOptions}
                  placeholder={t('launchModel.status')}
                  allowClear
                  className="w-40"
                />
              )}
            </div>
          </div>

          {loading ? (
            <div className="flex min-h-72 items-center justify-center text-muted-foreground">
              <Loader2 className="mr-2 size-5 animate-spin" />
              Loading models...
            </div>
          ) : visibleModels.length ? (
            <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
              {visibleModels.map(renderModelCard)}
            </div>
          ) : (
            <div className="flex min-h-72 flex-col items-center justify-center rounded-lg border border-dashed border-border text-center">
              <Info className="mb-3 size-8 text-muted-foreground" />
              <p className="font-medium">No models found</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Try changing your search or filters.
              </p>
            </div>
          )}
        </TabsContent>
      </Tabs>

      <ConfirmDialog
        isOpen={!!deleteModel}
        onOpenChange={(open) => !open && setDeleteModel(undefined)}
        description={t('launchModel.confirmDeleteCustomModel', {
          modelName: deleteModel?.model_name,
        })}
        confirmText={t('common.confirm')}
        confirmClassName="bg-destructive  hover:bg-destructive/90"
        onConfirm={handleDeleteModel}
      />
      <LaunchDialog
        model={selectedModel}
        modelType={requestType}
        gpuAvailable={gpuAvailable}
        onOpenChange={(open) => !open && setSelectedModel(undefined)}
      />
    </PageContainer>
  );
};

export default LaunchModel;
