classdef BinaryTree
    
    properties
        m_idxs
    end
    methods
        % -----------------------------------------------------------------
        function T = BinaryTree(m_idxs)
            if nargin == 1
                T.m_idxs = m_idxs;
            else
                T.m_idxs = zeros(0,2);
            end
        end
        % -----------------------------------------------------------------
        function nc = n_nodes(self)
            nc = size(self.m_idxs,1);
        end
        % -----------------------------------------------------------------
        function margin = get_margin(self)
            margin = zeros(0,2);
            for i=1:self.n_nodes
                level = self.m_idxs(i,1);
                index = self.m_idxs(i,2);
                child1 = [level+1, 2*index];
                child2 = [level+1, 2*index+1];
                if ~ismember(child1, self.m_idxs, 'rows')
                    margin = [margin; child1];
                end
                if ~ismember(child2, self.m_idxs, 'rows')
                    margin = [margin; child2];
                end
            end
        end
        % -----------------------------------------------------------------
        function self = add_node(self, new_midx)
            self.m_idxs = [self.m_idxs; new_midx];
        end
        % -----------------------------------------------------------------
    end
    
%     properties
%         root
%         midx
%     end
%     
%     methods
%         % -----------------------------------------------------------------
%         function T = BinaryTree()
%            T.root = Node(0,0);
%         end
%         function margin = get_margin(self)
% %             if isempty(node.left_child) || isempty(node.right_child)
% %                 list = [list, node];
% %             end
% %             if ~isempty(node.left_child)
% %                 get_child_nodes(node.left_child, list)
% %             end
% %             if ~isempty(node.right_child)
% %                 get_child_nodes(node.right_child, list)
% %             end
%             margin = zeros(0,2);
%             for i=1:self.n_nodes
%                 level = self.m_idxs(i,1);
%                 index = self.m_idxs(i,2);
%                 child1 = [level+1, 2*index];
%                 child2 = [level+1, 2*index+1];
%                 if ~ismember(child1, self.m_idxs, 'rows')
%                     margin = [margin; child1];
%                 end
%                 if ~ismember(child2, self.m_idxs, 'rows')
%                     margin = [margin; child2];
%                 end
%             end
%         end


%         % -----------------------------------------------------------------
% %         function get_levels(node, list)
% %             if isempty(node.left_child) || isempty(node.right_child)
% %                 list = [list, node];
% %             end
% %             if ~isempty(node.left_child)
% %                 get_child_nodes(node.left_child, list)
% %             end
% %             if ~isempty(node.right_child)
% %                 get_child_nodes(node.right_child, list)
% %             end
% %         end
%         % -----------------------------------------------------------------
% %         function get_indices(self)
% %             self.get_indices
% %         end
%         % -----------------------------------------------------------------
%         function list = get_children(self)
%             list = get_child_nodes(self.root, []);
%         end
%         % -----------------------------------------------------------------
%         function list = get_child_nodes(node, list)
%         % get children and assign to list
%             if isempty(node.left_child) || isempty(node.right_child)
%                 list = [list, node];
%             end
%             if ~isempty(node.left_child)
%                 get_child_nodes(node.left_child, list)
%             end
%             if ~isempty(node.right_child)
%                 get_child_nodes(node.right_child, list)
%             end
%         end
%         % -----------------------------------------------------------------
%         function list = get_margin(child_list)
%             midx = zeros(0,2);
%             for i=1:length(child_list)
%                 node = child_list(i);
%                 level = node.level;
%                 index = node.index;
%                 if isempty(node.left_child)
%                     midx = [midx; [level+1, 2*index]];
%                 end
%                 if isempty(node.right_child)
%                     midx = [midx; [level+1, 2*index+1]];
%                 end
%             end
%         end
%         % -----------------------------------------------------------------
%         % get children and assign to list
% %             if isempty(node.left_child) || isempty(node.right_child)
% %                 list = [list, node];
% %             end
% %             if ~isempty(node.left_child)
% %                 get_child_nodes(node.left_child, list)
% %             end
% %             if ~isempty(node.right_child)
% %                 get_child_nodes(node.right_child, list)
% %             end
% %         end
% % 
%         % -----------------------------------------------------------------
%         function self = add_child(self, node, index)
%             % find parent
%             parent_idx = 
%             
%             % add child
%             leaf_node.left_child = Node(level, index)
%             
%         end
%         % -----------------------------------------------------------------
%         function plot_tree(self)
%             lev = self.get_levels(self.root, []);
%             ind = self.get_indices(self.root, []);
%             G = graph(lev,ind);
%             plot(G,'LineWidth',2)
%         end
%         % -----------------------------------------------------------------
%     end
    
end