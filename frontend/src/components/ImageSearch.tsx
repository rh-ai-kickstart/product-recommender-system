import { useState } from 'react';
import {
  Button,
  InputGroup,
  TextInput,
  Tooltip,
  Split,
  SplitItem,
  EmptyState,
  EmptyStateBody,
  Title,
  Card,
  CardBody,
  ToggleGroup,
  ToggleGroupItem,
  FileUpload,
} from '@patternfly/react-core';
import { LinkIcon, UploadIcon } from '@patternfly/react-icons';
import {
  useProductSearchByImageLink,
  useProductSearchByImage,
} from '../hooks/useProducts';
import { GalleryView } from './Gallery';
import { GallerySkeleton } from './gallery-skeleton';

export const ImageSearch: React.FC = () => {
  const [searchType, setSearchType] = useState<'url' | 'file'>('url');
  const [imageUrl, setImageUrl] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [filename, setFilename] = useState('');
  const [urlSearchTrigger, setUrlSearchTrigger] = useState('');
  const [fileSearchTrigger, setFileSearchTrigger] = useState<File | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const {
    data: urlData,
    error: urlError,
    isLoading: urlLoading,
  } = useProductSearchByImageLink(urlSearchTrigger, 10, !!urlSearchTrigger);
  const {
    data: fileData,
    error: fileError,
    isLoading: fileLoading,
  } = useProductSearchByImage(fileSearchTrigger, 10, !!fileSearchTrigger);

  // Use the appropriate data/error/loading based on search type
  const data = searchType === 'url' ? urlData : fileData;
  const error = searchType === 'url' ? urlError : fileError;
  const isLoading = searchType === 'url' ? urlLoading : fileLoading;

  const handleUrlSearch = async () => {
    if (!imageUrl.trim()) return;
    setIsSearching(true);
    setFileSearchTrigger(null); // Clear file search
    setUrlSearchTrigger(imageUrl);
    setIsSearching(false);
  };

  const handleFileSearch = async () => {
    if (!imageFile) return;
    setIsSearching(true);
    setUrlSearchTrigger(''); // Clear URL search
    setFileSearchTrigger(imageFile);
    setIsSearching(false);
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && searchType === 'url') {
      handleUrlSearch();
    }
  };

  const handleFileChange = (_event: any, file: File) => {
    setImageFile(file);
    setFilename(file.name);
  };

  const handleFileClear = () => {
    setImageFile(null);
    setFilename('');
    setFileSearchTrigger(null);
  };

  return (
    <div>
      <Card>
        <CardBody>
          <ToggleGroup aria-label='Search type selection'>
            <ToggleGroupItem
              text='Image URL'
              isSelected={searchType === 'url'}
              onChange={() => setSearchType('url')}
            />
            <ToggleGroupItem
              text='Upload Image'
              isSelected={searchType === 'file'}
              onChange={() => setSearchType('file')}
            />
          </ToggleGroup>
        </CardBody>
      </Card>

      {searchType === 'url' ? (
        <Split hasGutter style={{ marginTop: 16 }}>
          <SplitItem isFilled>
            <InputGroup>
              <TextInput
                value={imageUrl}
                onChange={(_event, value) => setImageUrl(value)}
                onKeyPress={handleKeyPress}
                placeholder='Enter image URL to find similar products'
                type='url'
                aria-label='Image URL input'
              />
              <Tooltip content='Search by image URL'>
                <Button
                  variant='control'
                  onClick={handleUrlSearch}
                  isDisabled={!imageUrl.trim() || isSearching}
                  icon={<LinkIcon />}
                >
                  Search Similar
                </Button>
              </Tooltip>
            </InputGroup>
          </SplitItem>
        </Split>
      ) : (
        <div style={{ marginTop: 16 }}>
          <FileUpload
            id='image-file-upload'
            value={imageFile || undefined}
            filename={filename}
            filenamePlaceholder='Drag and drop an image file or upload one'
            onFileInputChange={handleFileChange}
            onClearClick={handleFileClear}
            browseButtonText='Upload'
            accept='image/*'
            allowEditingUploadedText={false}
          />
          {imageFile && (
            <Button
              variant='primary'
              onClick={handleFileSearch}
              isDisabled={isSearching}
              icon={<UploadIcon />}
              style={{ marginTop: 8 }}
            >
              Search Similar
            </Button>
          )}
        </div>
      )}

      {/* Display results or loading state */}
      {isLoading || isSearching ? (
        <div style={{ marginTop: 20 }}>
          <GallerySkeleton count={8} />
        </div>
      ) : error ? (
        <EmptyState>
          <Title headingLevel='h4' size='lg'>
            Error searching for similar products
          </Title>
          <EmptyStateBody>
            There was an error while searching. Please try again.
            {error instanceof Error && <div>{error.message}</div>}
          </EmptyStateBody>
        </EmptyState>
      ) : data && data.length > 0 ? (
        <div style={{ marginTop: 20 }}>
          <Title headingLevel='h2' size='xl'>
            Similar Products
          </Title>
          <GalleryView products={data} />
        </div>
      ) : urlSearchTrigger || fileSearchTrigger ? (
        <EmptyState>
          <Title headingLevel='h4' size='lg'>
            No similar products found
          </Title>
          <EmptyStateBody>
            Try a different image {searchType === 'url' ? 'URL' : 'file'} or
            search using text instead.
          </EmptyStateBody>
        </EmptyState>
      ) : null}
    </div>
  );
};
