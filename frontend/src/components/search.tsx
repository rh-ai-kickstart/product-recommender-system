import { useState } from 'react';
import { SearchInput } from '@patternfly/react-core';
import { useNavigate } from '@tanstack/react-router';

export const Search: React.FunctionComponent = () => {
  const [value, setValue] = useState('');
  const navigate = useNavigate();

  const onChange = (value: string) => {
    setValue(value);
  };

  const onSearch = (_event: any, value: string) => {
    if (value.trim()) {
      navigate({ to: '/search', search: { q: value.trim() } });
    }
  };

  return (
    <SearchInput
      placeholder='Find a product'
      value={value}
      onChange={(_event, value) => onChange(value)}
      onSearch={onSearch}
      onClear={() => onChange('')}
      className='pf-v6-u-w-100'
    />
  );
};
