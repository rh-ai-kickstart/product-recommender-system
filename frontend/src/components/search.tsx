import { useState } from 'react';
import {
  SearchInput,
  Tabs,
  Tab,
  TabTitleText,
  TabContent,
  TabContentBody,
} from '@patternfly/react-core';
import { useNavigate } from '@tanstack/react-router';
import { ImageSearch } from './ImageSearch';

export const Search: React.FunctionComponent = () => {
  const [value, setValue] = useState('');
  const [activeTabKey, setActiveTabKey] = useState<string | number>(0);
  const navigate = useNavigate();

  const onChange = (value: string) => {
    setValue(value);
  };

  const onSearch = (_event: any, value: string) => {
    if (value.trim()) {
      navigate({ to: '/search', search: { q: value.trim() } });
    }
  };

  const handleTabClick = (_: any, tabIndex: string | number) => {
    setActiveTabKey(tabIndex);
  };

  return (
    <Tabs
      activeKey={activeTabKey}
      onSelect={handleTabClick}
      isBox
      aria-label='Search options'
      style={{ marginBottom: '20px' }}
    >
      <Tab eventKey={0} title={<TabTitleText>Text Search</TabTitleText>}>
        <TabContent id='text-search-tab'>
          <TabContentBody>
            <SearchInput
              placeholder='Find a product'
              value={value}
              onChange={(_event, value) => onChange(value)}
              onSearch={onSearch}
              onClear={() => onChange('')}
              className='pf-v6-u-w-100'
            />
          </TabContentBody>
        </TabContent>
      </Tab>
      <Tab eventKey={1} title={<TabTitleText>Image URL Search</TabTitleText>}>
        <TabContent id='image-search-tab'>
          <TabContentBody>
            <ImageSearch />
          </TabContentBody>
        </TabContent>
      </Tab>
    </Tabs>
  );
};
