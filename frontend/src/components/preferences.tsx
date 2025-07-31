import {
  ActionGroup,
  Button,
  Card,
  CardHeader,
  CardTitle,
  Flex,
  FlexItem,
  Gallery,
  GalleryItem,
  Skeleton,
} from '@patternfly/react-core';
import { useState } from 'react';
import { usePreferences, useSetPreferences } from '../hooks';

export function PreferencePage() {
  const [selected, setSelected] = useState<string[]>([]);
  const preferencesMutation = useSetPreferences();
  const [errorMessage, setErrorMessage] = useState('');

  const { data, isLoading, isError } = usePreferences();

  const handleCancel = () => {
    setSelected([]);
  };

  // const handleSubmit = useMutation<string, void>({
  //   mutationFn: async () => {
  //     // Ensure addPreferences returns a Promise<string>
  //     const formattedPreferences = selected.join("|");
  //     return await addPreferences(formattedPreferences);
  //   },
  //   onSuccess: async () => {
  //     void queryClient.invalidateQueries({ queryKey: ["new-preferences"] });
  //     setSelected([]);
  //     console.log("Preferences added successfully");
  //     navigate({ to: "/" });
  //   },
  //   onError: (error) => {
  //     console.error("Error adding preferences:", error);
  //   },
  // });

  const handleSubmit = async (
    event: React.MouseEvent<HTMLButtonElement, MouseEvent>
  ) => {
    event.preventDefault();

    try {
      setErrorMessage('');
      console.log(selected.join('|'));
      await preferencesMutation.mutateAsync({
        preferences: selected.join('|'),
      });
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Preferences failed to load. Please try again.'
      );
    }
  };

  return (
    <>
      {isLoading ? (
        <Skeleton style={{ height: 200, width: '100%' }} />
      ) : isError ? (
        <div>Error fetching preferences</div>
      ) : (
        <>
          <Gallery hasGutter style={{ width: '100%' }}>
            {data?.map(category => (
              <GalleryItem key={category}>
                <Card
                  aria-label={`Select ${category}`}
                  isSelectable
                  isSelected={selected.includes(category)}
                  onClick={() => {
                    if (selected.includes(category)) {
                      setSelected(selected.filter(item => item !== category));
                    } else {
                      setSelected([...selected, category]);
                    }
                  }}
                  style={{
                    minWidth: 250,
                    cursor: 'pointer',
                    backgroundColor: selected.includes(category)
                      ? '#e7f1ff'
                      : 'white',
                    border: selected.includes(category)
                      ? '2px solid #0066cc'
                      : '1px solid #d2d2d2',
                  }}
                  key={category}
                >
                  <CardTitle>{category}</CardTitle>
                </Card>
              </GalleryItem>
            ))}
          </Gallery>
          <Flex
            style={{ marginTop: 24 }}
            justifyContent={{ default: 'justifyContentFlexEnd' }}
          >
            <FlexItem>
              <ActionGroup>
                <Button variant='primary' type='submit' onClick={handleSubmit}>
                  Submit
                </Button>
                <Button variant='link' onClick={handleCancel}>
                  Cancel
                </Button>
              </ActionGroup>
            </FlexItem>
          </Flex>
        </>
      )}
    </>
  );
}
