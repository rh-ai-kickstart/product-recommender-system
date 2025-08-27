import { useState } from 'react';
import {
  SearchInput,
  Tabs,
  Tab,
  TabTitleText,
  TabContent,
  TabContentBody,
  Card,
  CardBody,
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
    <Card
      style={{
        borderRadius: '12px',
        border: '1px solid #dee2e6',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      }}
    >
      <CardBody style={{ padding: '16px' }}>
        <Tabs
          activeKey={activeTabKey}
          onSelect={handleTabClick}
          isBox={false}
          aria-label='Search options'
          style={{
            marginBottom: '16px',
            borderRadius: '8px',
          }}
        >
          <Tab
            eventKey={0}
            title={
              <TabTitleText style={{ fontWeight: '600' }}>
                Text Search
              </TabTitleText>
            }
            style={{
              borderRadius: '8px 8px 0 0',
            }}
          >
            <TabContent id='text-search-tab'>
              <TabContentBody style={{ padding: '16px 0' }}>
                <SearchInput
                  placeholder='Find a product by name, brand, or description'
                  value={value}
                  onChange={(_event, value) => onChange(value)}
                  onSearch={onSearch}
                  onClear={() => onChange('')}
                  className='pf-v6-u-w-100'
                  style={{
                    borderRadius: '8px',
                    border: '1px solid #ced4da',
                    fontSize: '14px',
                  }}
                />
              </TabContentBody>
            </TabContent>
          </Tab>
          <Tab
            eventKey={1}
            title={
              <TabTitleText style={{ fontWeight: '600' }}>
                Visual Search
              </TabTitleText>
            }
            style={{
              borderRadius: '8px 8px 0 0',
            }}
          >
            <TabContent id='image-search-tab'>
              <TabContentBody style={{ padding: '16px 0' }}>
                <ImageSearch />
              </TabContentBody>
            </TabContent>
          </Tab>
        </Tabs>
      </CardBody>
    </Card>
  );
};