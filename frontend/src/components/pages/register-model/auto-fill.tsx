'use client';

import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useI18n } from '@/contexts/i18n-context';

const AutoFill = ()=>{
  const { t } = useI18n();
  return (
   <>
     <Button className='mb-1'><Sparkles size={14}/>{t('registerModel.autoFill')}</Button>

   </>
  )
}
export default AutoFill;