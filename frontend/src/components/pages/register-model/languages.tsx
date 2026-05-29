'use client';

import { FC } from 'react';

import { CheckboxGroup } from '@/components/ui/checkbox-group';
import { MultiSelect } from '@/components/ui/multi-select';

import { LANGUAGES_OPTIONS, LANGUAGES_CHECKBOX_OPTIONS } from '@/constants/register';

import type { BaseFormFieldProps } from '@/types/form';

interface LanguagesProps extends BaseFormFieldProps<string[]> {}

const CHECKBOX_VALUES = new Set(LANGUAGES_CHECKBOX_OPTIONS.map((item) => item.value));

const Languages: FC<LanguagesProps> = ({ value, onChange, error, placeholder }) => {
  const selectedValues = Array.isArray(value) ? value : [];

  const checkboxValue = selectedValues.filter((item) => CHECKBOX_VALUES.has(item));

  const selectValue = selectedValues.filter((item) => !CHECKBOX_VALUES.has(item));

  const handleCheckBoxChange = (values: string[]) => {
    onChange?.([...values, ...selectValue]);
  };

  const handleSelectChange = (values: string[]) => {
    onChange?.([...checkboxValue, ...values]);
  };

  return (
    <div className="flex items-center gap-2">
      <CheckboxGroup
        className="shrink-0"
        error={error}
        options={LANGUAGES_CHECKBOX_OPTIONS}
        value={checkboxValue}
        onChange={handleCheckBoxChange}
      />

      <MultiSelect
        className="flex-1"
        error={error}
        placeholder={placeholder}
        value={selectValue}
        onChange={handleSelectChange}
        options={LANGUAGES_OPTIONS}
      />
    </div>
  );
};

export default Languages;
