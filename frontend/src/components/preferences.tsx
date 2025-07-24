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
} from "@patternfly/react-core";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { addPreferences, fetchNewPreferences } from "../services/products";
import { useNavigate } from "@tanstack/react-router";

export function PreferencePage() {
  const [selected, setSelected] = useState<string[]>([]);
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const { data, isError, isLoading } = useQuery({
    queryKey: ["new-preferences"], // A unique key for this query
    queryFn: fetchNewPreferences, // The async function to fetch data
  });

  const handleCancel = () => {
    setSelected([]);
  };

  const handleSubmit = useMutation<string, void>({
    mutationFn: async () => {
      // Ensure addPreferences returns a Promise<string>
      const formattedPreferences = selected.join("|");
      return await addPreferences(formattedPreferences);
    },
    onSuccess: async () => {
      void queryClient.invalidateQueries({ queryKey: ["new-preferences"] });
      setSelected([]);
      console.log("Preferences added successfully");
      navigate({ to: "/" });
    },
    onError: (error) => {
      console.error("Error adding preferences:", error);
    },
  });

  return (
    <>
      {isLoading ? (
        <Skeleton style={{ height: 200, width: "100%" }} />
      ) : isError ? (
        <div>Error fetching preferences</div>
      ) : (
        <>
          <Gallery hasGutter style={{ width: "100%" }}>
            {data?.map((category) => (
              <GalleryItem key={category}>
                <Card
                  isSelectable
                  isSelected={selected.includes(category)}
                  onClick={() => {
                    if (selected.includes(category)) {
                      setSelected(selected.filter((item) => item !== category));
                    } else {
                      setSelected([...selected, category]);
                    }
                  }}
                  style={{
                    minWidth: 250,
                    cursor: "pointer",
                    backgroundColor: selected.includes(category)
                      ? "#e7f1ff"
                      : "white",
                    border: selected.includes(category)
                      ? "2px solid #0066cc"
                      : "1px solid #d2d2d2",
                  }}
                  key={category}
                >
                  <CardHeader>
                    <CardTitle>{category}</CardTitle>
                  </CardHeader>
                </Card>
              </GalleryItem>
            ))}
          </Gallery>
          <Flex style={{ marginTop: 24 }} justifyContent={{ default: "justifyContentFlexEnd" }}>
            <FlexItem>
              <ActionGroup>
                <Button
                  variant="primary"
                  type="submit"
                  onClick={() => handleSubmit.mutate()}
                >
                  Submit
                </Button>
                <Button variant="link" onClick={handleCancel}>
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
