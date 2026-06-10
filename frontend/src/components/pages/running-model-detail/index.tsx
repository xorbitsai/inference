'use client';

import { FC, useEffect, useState } from 'react';

import request from '@/lib/request';
import type { RunningModelDetail } from '@/types/services';
interface RunningModelDetailProps {
  modelUid: string;
}
const RunningModelDetail: FC<RunningModelDetailProps> = ({ modelUid }) => {
  const [data, setData] = useState<RunningModelDetail>({});
  const fetchDetail = () => {
    request.get(`/v1/models/${modelUid}`).then((res) => setData(res));
  };
  useEffect(() => fetchDetail(), []);
  return <div />;
};
export default RunningModelDetail;
