export interface BaseFormFieldProps<T = any> {
  value?: T;

  onChange?: (value: T) => void;

  error?: boolean;

  placeholder?: string;

  disabled?: boolean;

  className?: string;
}

export type NamePath = Array<string | number>;

export type FieldName = string | number | NamePath;

export type FormValues = Record<string, any>;

type RequiredRule = {
  required: boolean;

  message?: string;

  pattern?: never;

  validator?: never;
};

type PatternRule = {
  pattern: RegExp;

  message: string;

  required?: never;

  validator?: never;
};

type ValidatorRule = {
  validator: (value: any) => boolean;

  message: string;

  required?: never;

  pattern?: never;
};

export type Rule = RequiredRule | PatternRule | ValidatorRule;

export interface FormInstance {
  store: React.MutableRefObject<FormValues>;

  initialValues: React.MutableRefObject<FormValues>;

  getFieldValue: (name: FieldName) => any;

  getInitialFieldValue: (name: FieldName) => any;

  hasFieldValue: (name: FieldName) => boolean;

  getFieldsValue: () => FormValues;

  initFieldValue: (name: FieldName) => void;

  deleteFieldValue: (name: FieldName) => void;

  mountField: (name: FieldName) => () => void;

  setFieldValue: (name: FieldName, value: any) => void;

  setFieldsValue: (values: Partial<FormValues>) => void;

  resetFields: () => void;

  subscribe: (callback: () => void) => () => void;

  registerReset: (callback: () => void) => () => void;
}

export type FormContextType = {
  form: FormInstance;

  errors: Record<string, string>;

  touched: Record<string, boolean>;

  setFieldError: (name: FieldName, error: string) => void;

  setFieldTouched: (name: FieldName, touched: boolean) => void;

  clearFieldState: (name: FieldName) => void;

  registerField: (name: FieldName, validate: () => string) => void;

  unregisterField: (name: FieldName) => void;
};

export interface FormFieldProps {
  name: FieldName;

  label?: React.ReactNode;

  extra?: React.ReactNode;

  rules?: Rule[];

  placeholder?: React.ReactNode;

  disabled?: boolean;

  children: React.ReactElement;

  valuePropName?: string;

  layout?: 'vertical' | 'horizontal';

  className?: string;

  normalize?: (value: any) => any;
}

export interface FormListField {
  key: string;

  name: number;
}

export interface FormListRenderProps<T = any> {
  fields: FormListField[];

  add: (defaultValue?: T) => void;

  remove: (index: number) => void;
}

export interface FormListProps<T = any> {
  name: FieldName;

  label?: React.ReactNode;

  extra?: React.ReactNode;

  layout?: 'vertical' | 'horizontal';

  className?: string;

  children: (params: FormListRenderProps<T>) => React.ReactElement;

  renderAction?: (params: FormListRenderProps<T>) => React.ReactNode;
}
