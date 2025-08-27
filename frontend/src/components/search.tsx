import { useState } from 'react';
import {
  SearchInput,
  Tabs,
  Tab,
  TabTitleText,
  TabContent,
  TabContentBody,
  Title,
} from '@patternfly/react-core';
import { SearchIcon, ImageIcon } from '@patternfly/react-icons';
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
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 24px' }}>
      <Title
        headingLevel='h1'
        size='2xl'
        style={{
          textAlign: 'center',
          marginBottom: '32px',
          color: '#2c3e50',
          fontWeight: '700',
        }}
      >
        Product Search
      </Title>

      <Tabs
        activeKey={activeTabKey}
        onSelect={handleTabClick}
        isBox
        aria-label='Search options'
        style={{
          marginBottom: '32px',
          background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
          borderRadius: '12px',
          padding: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          border: 'none',
        }}
      >
        <Tab
          eventKey={0}
          title={
            <TabTitleText>
              <SearchIcon style={{ marginRight: '8px' }} />
              Text Search
            </TabTitleText>
          }
          style={{
            borderRadius: '8px',
            ...(activeTabKey === 0 && {
              background: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
              color: 'white',
            }),
          }}
        >
          <TabContent id='text-search-tab'>
            <TabContentBody style={{ padding: '24px 0' }}>
              <SearchInput
                placeholder='Search for products by name, description, or category...'
                value={value}
                onChange={(_event, value) => onChange(value)}
                onSearch={onSearch}
                onClear={() => onChange('')}
                className='pf-v6-u-w-100'
                style={{
                  fontSize: '16px',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  border: '2px solid #e9ecef',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                  transition: 'all 0.2s ease',
                }}
              />
            </TabContentBody>
          </TabContent>
        </Tab>
        <Tab
          eventKey={1}
          title={
            <TabTitleText>
              <ImageIcon style={{ marginRight: '8px' }} />
              Visual Search
            </TabTitleText>
          }
          style={{
            borderRadius: '8px',
            ...(activeTabKey === 1 && {
              background: 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
              color: 'white',
            }),
          }}
        >
          <TabContent id='image-search-tab'>
            <TabContentBody style={{ padding: '24px 0' }}>
              <ImageSearch />
            </TabContentBody>
          </TabContent>
        </Tab>
      </Tabs>
    </div>
  );
};
